import argparse
from collections import OrderedDict
import functools
import gc
import logging
import math
import numpy as np
import os
from packaging import version
from pathlib import Path
from PIL import Image
import random
import shutil
from typing import List, Optional, Tuple, Union

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.utils.checkpoint
from torch.utils.data import DataLoader, default_collate
from torchvision import models, transforms
from torchvision.utils import save_image
import torchvision.transforms as transforms
import torchvision.transforms.functional as TF
from torchvision.transforms.functional import to_pil_image
from models.sampler import ImbalancedDatasetSampler

import wandb
from accelerate import Accelerator
from accelerate.logging import get_logger
from accelerate.utils import ProjectConfiguration, set_seed
from tqdm import tqdm
import transformers

import diffusers
from diffusers.utils.torch_utils import randn_tensor
from diffusers.optimization import get_scheduler
from diffusers.schedulers import LCMScheduler

from config import *
from dataset import CustomImageDataset
from models.attention import SpatialTransformer, AttentionBlock
from models.ddpm_scheduler import NoiseScheduler
from models.embedding import TimeEmbedding, ConditionalEmbedding, LabelEmbedding
from models.engine import DDIMSampler, DDIMSamplerEncoder, ConsisctencySampler
from models.unet import Swish, UpSample, DownSample, AdaGroupNorm, TimestepEmbedSequential, ResBlock, ResBlockCond, ResBlockAdaGN, ResBlockImageClassConcat
from models.unet import UNet
from utils import DDIMSolver, update_ema, guidance_scale_embedding, append_dims, extract, set_seed, load_checkpoint
from utils import scalings_for_boundary_conditions, get_predicted_original_sample, get_predicted_noise, get_ddim_timesteps
from utils_ccdm import get_model

logger = get_logger(__name__)


def ArgsStringToDict(args_):
    """  ['0.25:0.6', '0.5:0.3', '0.75:0.1'] to {0.25: 0.6, 0.5: 0.3, 0.75: 0.1} """
    segment_dict = {}
    for segment in args_:
        key, value = segment.split(":")
        segment_dict[float(key)] = float(value)
    return segment_dict


def run_eval_and_log(model, noise_scheduler, CFG, args, accelerator, epoch, weight_dtype=torch.float32, name="target"):
    logger.info("Running validation... ")
    num_atr, num_obj = args.dataset_nums_cond[0], args.dataset_nums_cond[1]
    n_samples =  num_atr * num_obj

    sampler = ConsisctencySampler(model, noise_scheduler, n_samples, args, device=accelerator.device)

    if args.seed is None:
        generator = None
    else:
        generator = torch.Generator(device=device).manual_seed(args.seed)        

    # create conditions of each class
    # create conditions like [0,0,0,1,1,1, ...] [0,1,2,3,0,1,2,3, ...]
    c1 = torch.arange(0, num_atr)
    c2 = torch.arange(0, num_obj)
    c1 = c1.repeat(n_samples // num_atr, 1).permute(1, 0).reshape(-1)
    c2 = c2.repeat(n_samples // num_obj)
    c1, c2 = c1.to(accelerator.device), c2.to(accelerator.device)

    # create noise
    x_t = torch.randn(n_samples, 3, args.img_size, args.img_size).to(device=accelerator.device, dtype=torch.float32)

    images = sampler(
                c1, 
                c2,
                x_t = x_t, 
                num_inference_steps = 4,
                guidance_scale = 2.8,
                generator=generator
             )

    # save image
    os.makedirs(os.path.join('result', args.exp, args.save_path), exist_ok=True)
    save_image(images, os.path.join('result', args.exp, args.save_path, f'epoch_{epoch}.png'))
    
            
    # log image
    images = images.permute(0, 2, 3, 1)
    images = images.cpu().detach().numpy()
    c1, c2 = c1.cpu().detach().numpy(), c2.cpu().detach().numpy()
    
    images = [(f"{CFG.IDX2ATR[c1[i]]} {CFG.IDX2OBJ[c2[i]]}", images[i, :, :, :]) for i in range(n_samples)]
    if not args.debug:
        accelerator.log({f"{name} model eval": [wandb.Image(img, caption=label) for label, img in images]})

    del sampler
    gc.collect()
    torch.cuda.empty_cache()

    return images
    
    
def main(args):
    
    logging_dir = Path(args.save_path, args.logging_dir)
    # accelerator_project_config = ProjectConfiguration(project_dir=args.save_path, logging_dir=logging_dir)   
    
    accelerator = Accelerator(log_with="wandb")
    # Make one log on every process with the configuration for debugging.
    logging.basicConfig(
        format="%(asctime)s - %(levelname)s - %(name)s - %(message)s",
        datefmt="%m/%d/%Y %H:%M:%S",
        level=logging.INFO,
    )
    logger.info(accelerator.state, main_process_only=False)
    if accelerator.is_local_main_process:
        transformers.utils.logging.set_verbosity_warning()
        diffusers.utils.logging.set_verbosity_info()
    else:
        transformers.utils.logging.set_verbosity_error()
        diffusers.utils.logging.set_verbosity_error()    
    
    if args.seed is not None:
        set_seed(args.seed)
        
    if args.dataset_nums_cond is None:
        args.dataset_nums_cond = args.pretrained_nums_cond
        
    if accelerator.is_main_process:
        if args.debug:
            break
            
        tracker_config = {
            "architecture": args.arch,
            "dataset": args.data.split('/')[-1],
            "stepfuse_method": args.stepfuse_method,
            "img_size": args.img_size,
            "train_batch_size": args.train_batch_size,
            "learning_rate": args.lr,
            "epochs": args.epochs,
            "dataset_nums_cond": args.dataset_nums_cond,
            "pretrained_nums_cond": args.pretrained_nums_cond,
            "guidance_scale_interval": args.w_interval,
            "num_ddim_timesteps": args.num_ddim_timesteps
        }
        accelerator.init_trackers(args.exp, config=tracker_config)
        
    # 1.Create the noise scheduler and the desired noise schedule.     
    # ddpm NoiseScheduler calculates the alpha and sigma noise schedules (based on the alpha bars) for us
    noise_scheduler = NoiseScheduler()
    
    # Initialize the DDIM ODE solver for distillation.
    solver = DDIMSolver(
            noise_scheduler.alphas_cumprod.numpy(),
            timesteps=noise_scheduler.config.num_train_timesteps,
            ddim_timesteps=args.num_ddim_timesteps,
            )
        
    # 2-1.Load compositional conditional teacher U-Net
    teacher_unet = get_model(args)
    # checkpoint = torch.load(r"checkpoints/colored_mnist/mnist5x10x64_ic.pth",map_location='cpu')["model"]
    checkpoint = torch.load(os.path.join('checkpoints', args.exp, args.save_path, args.pretrained_pth), map_location='cpu')["model"]

    new_ckpt = OrderedDict()
    for k, v in checkpoint.items():
        if k.startswith("module"):
            new_ckpt[k[7:]] = v
        else:
            new_ckpt[k] = v

    # Load model parameters (with strict=False to allow some missing or extra keys)
    missing_keys, unexpected_keys = teacher_unet.load_state_dict(new_ckpt, strict=False)
    print("Teacher Missing keys:", missing_keys)
    print("Teacehr Unexpected keys:", unexpected_keys)

    # Manually assign weights for special cases
    for i in range(len(missing_keys)):
        assert missing_keys[i].split(".")[-1] == unexpected_keys[i].split(".")[-1]
        teacher_unet.state_dict()[missing_keys[i]].copy_(new_ckpt[unexpected_keys[i]])

    # Freeze teacher_unet
    teacher_unet.requires_grad_(False)
    
    # 2.2 Create online student U-Net. Will be updated by the optimizer (e.g. backprop)
    # Add `time_cond_proj_dim` to the student U-Net
    unet = get_model(args, args.time_cond_proj_dim)
    # load teacher_unet weights into unet
    unet.load_state_dict(teacher_unet.state_dict(), strict=False)
    unet.train()
    
    # 2.3 Create target student U-Net. This will be updated via EMA updates (polyak averaging).
    # Initialize from (online) unet
    target_unet = get_model(args, args.time_cond_proj_dim)
    target_unet.load_state_dict(unet.state_dict(), strict=True)
    target_unet.train()
    target_unet.requires_grad_(False)
    
    # 2.4 Move to gpu device
    unet.to(accelerator.device)
    target_unet.to(accelerator.device)
    teacher_unet.to(accelerator.device)
    noise_scheduler.to(accelerator.device)
    solver.to(accelerator.device)
    
    # Todo: Handle saving and loading of checkpoints
    ###
    
    # 3. Train data handling
    transform = transforms.Compose([
        transforms.Resize((args.img_size, args.img_size)),
        transforms.ToTensor(),
        transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
    ])
    train_ds = CustomImageDataset(root=args.data, transform=transform)
    balance_sampler = ImbalancedDatasetSampler(train_ds, labels=train_ds.labels)
    dataloader = DataLoader(train_ds, batch_size=args.train_batch_size, sampler=balance_sampler, num_workers=4)
    
    # condition indexing for dataset
    ds_type = args.data.split("/")[-1]
    if "ut" in ds_type :
        CFG = Zappo50K()
    elif "CelebA" in ds_type:
        CFG = CelebA()
    elif "mnist" in ds_type:
        CFG = Colored_MNIST()
    else:
        CFG = toy_dataset()
    
    # 4.1 Optimizer
    optimizer_class = torch.optim.AdamW
    optimizer = optimizer_class(
        unet.parameters(),
        lr=args.lr,
        betas=(args.adam_beta1, args.adam_beta2),
        weight_decay=args.adam_weight_decay,
        eps=args.adam_epsilon,
    )
    # 4.2 LR scheduler
    lr_scheduler = get_scheduler(
        args.lr_schedule,
        optimizer=optimizer,
        num_warmup_steps=args.lr_warmup_steps,
        num_training_steps=args.epochs*len(dataloader),
    )
    
    start_epoch = 0 
    
    # resume from checkpoint
    if args.resume_pth is not None:
        try:
            start_epoch = load_checkpoint(
                unet, 
                os.path.join('checkpoints', args.exp, args.save_path, args.resume_pth),
                optimizer=optimizer
            )
            _ = load_checkpoint(
                target_unet, 
                os.path.join('checkpoints', args.exp, args.save_path, args.resume_pth)
            )      
        except FileNotFoundError:
            print("No checkpoint found, starting training from scratch.")
            

    # 5. Prepare everything with our accelerator for training
    if args.resume_pth is not None:
        lr_scheduler.last_epoch = start_epoch
        
    unet, optimizer, lr_scheduler, dataloader = accelerator.prepare(unet, optimizer, lr_scheduler, dataloader)
    
    #num_update_steps_per_epoch = math.ceil(len(dataloader) / args.gradient_accumulation_steps)
    num_update_steps_per_epoch = len(dataloader)
    max_train_steps = args.epochs * num_update_steps_per_epoch
    global_step = start_epoch * num_update_steps_per_epoch
    best_loss = float('inf')
    
    progress_bar = tqdm(
        range(0, max_train_steps),
        initial=global_step,
        desc="Steps",
        # Only show the progress bar once on each machine.
        disable=not accelerator.is_local_main_process,
    )
    # progress_bar = tqdm(dataloader, desc=f'Epoch {epoch}')
    
    # 6. Train Loop!
    logger.info("***** Running training *****")
    logger.info(f"  Num batches each epoch = {len(dataloader)}")
    logger.info(f"  Num Epochs = {args.epochs}")
    logger.info(f"  Instantaneous batch size per device = {args.train_batch_size}")
    logger.info(f"  Total optimization steps = {max_train_steps}")
    
    for epoch in range(start_epoch+1, args.epochs+1):
        epoch_loss = 0.
        epoch_batches = 0
        for step, batch in enumerate(dataloader):
            # 1. Load and process the image and text conditioning
            x = batch["image"].to(accelerator.device)
            c1 = torch.tensor([CFG.ATR2IDX[a] for a in batch["atr"]], dtype=torch.long, device=accelerator.device)
            c2 = torch.tensor([CFG.OBJ2IDX[o] for o in batch["obj"]], dtype=torch.long, device=accelerator.device)
            bsz = x.shape[0]
            
            # 2. Sample a random timestep for each image t_n from the ODE solver timesteps without bias.
            # For the DDIM solver, the timestep schedule is [T - 1, T - k - 1, T - 2 * k - 1, ...]
            topk = noise_scheduler.config.num_train_timesteps // args.num_ddim_timesteps
            index = torch.randint(0, args.num_ddim_timesteps, (bsz,), device=x.device).long()
            start_timesteps = solver.ddim_timesteps[index]
            timesteps = start_timesteps - topk
            timesteps = torch.where(timesteps < 0, torch.zeros_like(timesteps), timesteps)
            
            # 3. Get boundary scalings for start_timesteps and (end) timesteps.
            c_skip_start, c_out_start = scalings_for_boundary_conditions(
                start_timesteps, timestep_scaling=args.timestep_scaling_factor
            )
            c_skip_start, c_out_start = [append_dims(c, x.ndim) for c in [c_skip_start, c_out_start]]
            c_skip, c_out = scalings_for_boundary_conditions(
                timesteps, timestep_scaling=args.timestep_scaling_factor
            )
            c_skip, c_out = [append_dims(c, x.ndim) for c in [c_skip, c_out]]
            
            # 4. Sample noise from the prior and add it to the latents according to the noise magnitude at each
            # timestep (this is the forward diffusion process) [z_{t_{n + k}} in Algorithm 1]
            noisy_model_input, epsilon = noise_scheduler.add_noise(x, start_timesteps, return_epsilon=True)
            
            # 5. Sample a random guidance scale w from U[w_min, w_max] and embed it
            w = torch.empty(bsz).uniform_(args.w_interval[0], args.w_interval[1])
            w_embedding = guidance_scale_embedding(w, embedding_dim=args.time_cond_proj_dim)
            w = w.reshape(bsz, 1, 1, 1)
            # Move to U-Net device and dtype
            w = w.to(device=x.device, dtype=x.dtype)
            w_embedding = w_embedding.to(device=x.device, dtype=x.dtype)
            
            # 6. Get online LCM prediction on z_{t_{n + k}} (noisy_model_input), w, c, t_{n + k} (start_timesteps)
            noise_pred = unet(
                noisy_model_input,
                start_timesteps,
                c1,
                c2,
                timestep_cond=w_embedding,
            )

            pred_x0 = get_predicted_original_sample(
                noise_pred,
                start_timesteps,
                noisy_model_input,
                noise_scheduler.config.prediction_type,
                noise_scheduler.alpha_schedule,
                noise_scheduler.sigma_schedule,
            )

            model_pred = c_skip_start * noisy_model_input + c_out_start * pred_x0
            
            # 7. Compute the conditional and unconditional teacher model predictions to get CFG estimates of the
            # predicted noise eps_0 and predicted original sample x_0, then run the ODE solver using these
            # estimates to predict the data point in the augmented PF-ODE trajectory corresponding to the next ODE
            # solver timestep.
            with torch.no_grad():
                # 7.1 Get teacher model prediction on noisy_model_input z_{t_{n + k}} and conditional embedding c
                cond_teacher_output = teacher_unet(
                    noisy_model_input,
                    start_timesteps,
                    c1,
                    c2,
                )
                cond_pred_x0 = get_predicted_original_sample(
                    cond_teacher_output,
                    start_timesteps,
                    noisy_model_input,
                    noise_scheduler.config.prediction_type,
                    noise_scheduler.alpha_schedule,
                    noise_scheduler.sigma_schedule,
                )
                cond_pred_noise = get_predicted_noise(
                    cond_teacher_output,
                    start_timesteps,
                    noisy_model_input,
                    noise_scheduler.config.prediction_type,
                    noise_scheduler.alpha_schedule,
                    noise_scheduler.sigma_schedule,
                )


                # 7.2 Get teacher model prediction on noisy_model_input z_{t_{n + k}} and unconditional embedding 0
                uncond_teacher_output = teacher_unet(
                    noisy_model_input,
                    start_timesteps,
                    c1,
                    c2,
                    force_drop_ids=True,
                )
                uncond_pred_x0 = get_predicted_original_sample(
                    uncond_teacher_output,
                    start_timesteps,
                    noisy_model_input,
                    noise_scheduler.config.prediction_type,
                    noise_scheduler.alpha_schedule,
                    noise_scheduler.sigma_schedule,
                )
                uncond_pred_noise = get_predicted_noise(
                    uncond_teacher_output,
                    start_timesteps,
                    noisy_model_input,
                    noise_scheduler.config.prediction_type,
                    noise_scheduler.alpha_schedule,
                    noise_scheduler.sigma_schedule,
                )

                # 7.3 Calculate the CFG estimate of x_0 (pred_x0) and eps_0 (pred_noise)
                # Note that this uses the LCM paper's CFG formulation rather than the Imagen CFG formulation
                pred_x0 = cond_pred_x0 + w * (cond_pred_x0 - uncond_pred_x0)
                pred_noise = cond_pred_noise + w * (cond_pred_noise - uncond_pred_noise)
                # 7.4 Run one step of the ODE solver to estimate the next point x_prev on the
                # augmented PF-ODE trajectory (solving backward in time)
                # Note that the DDIM step depends on both the predicted x_0 and source noise eps_0.
                x_prev_teacher = solver.ddim_step(pred_x0, pred_noise, index)
                
                # new! 7.5 adaptive x_prev.
                # use teacher model's prediction more in early epochs, ODE more in late epochs.
                x_prev_ode = noise_scheduler.add_noise(x, timesteps, epsilon=epsilon) 
                
                # args.stepfuse_method == "only_teacher":
                c_t = 1
                c_ode = 0
                
                if args.stepfuse_method == "piecewise_linear":
                    piecewise_dict = ArgsStringToDict(args.piecewise_segments)
                    # args.piecewise_segments may look like ["0.2:0.6", "0.5:0.3", "0.7:0.2", "0.9:0.1"]
                    # and c_t will be 1 at the beginning of the training epoch, 
                    # decreasing gradually to 0.6 at 20% of the epochs,
                    # 0.3 at 50% of the epochs, 0.2 at 70% of the epochs, 0.1 at 90%, finally 0 at last epochs.
                    
                    segments_keys = sorted(piecewise_dict.keys())
                    segments_values = [piecewise_dict[k] for k in segments_keys]
                    # Add start (1.0) and end (0.0) points
                    segments_keys = [0.0] + segments_keys + [1.0]
                    segments_values = [1.0] + segments_values + [0.0]
                    
                    # Compute the current progress ratio
                    progress = (epoch - (start_epoch+1)) / max_epoch
                    # Find the current segment
                    for i in range(len(segments_keys) - 1):
                        if segments_keys[i] <= progress < segments_keys[i + 1]:
                            # Compute linear interpolation in the current segment
                            c_t = segments_values[i] + (segments_values[i + 1] - segments_values[i]) * \
                                  (progress - segments_keys[i]) / (segments_keys[i + 1] - segments_keys[i])
                            c_ode = 1 - c_t
                            break   
                
                # elif args.stepfuse_method == "sigmoid":
                    # current_epochs = np.arange(total_epochs)
                    # k = 0.05  # Steepness of the curve
                    # m = total_epochs // 1.6  # Midpoint of the transition
                    # c_start = 1  # Initial value of c_ode
                    # c_end = 0.1  # Final value of c_ode

                    # # Sigmoid-based curve
                    # curve = 1 / (1 + np.exp(-k * (current_epochs - m)))
                    # c_ode = c_end + (c_start - c_end) * curve
                    # c_t = 1 - c_ode 
                    
                elif args.stepfuse_method == "only_ode":
                    c_ode = 1
                    c_t = 0
                    
                # elif args.stepfuse_method == "only_teacher":    
                #    c_ode = 0
                #    c_t = 1                  
                    
                x_prev = c_t * x_prev_teacher + c_ode * x_prev_ode
            
            # 8. Get target LCM prediction on x_prev, w, c, t_n (timesteps)
            with torch.no_grad():
                target_noise_pred = target_unet(
                    x_prev.float(),
                    timesteps,
                    c1,
                    c2,
                    timestep_cond=w_embedding,
                )

                pred_x0 = get_predicted_original_sample(
                    target_noise_pred,
                    timesteps,
                    x_prev,
                    noise_scheduler.config.prediction_type,
                    noise_scheduler.alpha_schedule,
                    noise_scheduler.sigma_schedule,
                )

                target = c_skip * x_prev + c_out * pred_x0
                
            # 9. Calculate loss
            if args.loss_type == "l2":
                loss = F.mse_loss(model_pred.float(), target.float(), reduction="mean")
            elif args.loss_type == "huber":
                loss = torch.mean(
                    torch.sqrt((model_pred.float() - target.float()) ** 2 + args.huber_c**2) - args.huber_c
                )
                
            # 10. Backpropagate on the online student model (`unet`)
            accelerator.backward(loss)
            if accelerator.sync_gradients:
                accelerator.clip_grad_norm_(unet.parameters(), 1.0)
            optimizer.step()
            lr_scheduler.step()
            optimizer.zero_grad(set_to_none=True)
            
            epoch_batches += 1
            epoch_loss += loss.item()
            
            # Checks if the accelerator has performed an optimization step behind the scenes
            if accelerator.sync_gradients:
            # 11. Make EMA update to target student model parameters (`target_unet`)
                update_ema(target_unet.parameters(), unet.parameters(), args.ema_decay)
                progress_bar.update(1)
                global_step += 1
                
                if accelerator.is_main_process:
                    save_root = os.path.join('checkpoints', args.exp, args.save_path)
                    os.makedirs(save_root, exist_ok=True)
                    avg_loss = epoch_loss / epoch_batches
                    
                    # log evaluation images to wandb
                    # if global_step % args.eval_interval == 0:
                    if epoch % args.eval_interval == 0:
                        run_eval_and_log(unet, noise_scheduler, CFG, args, accelerator, epoch, "online")
                        run_eval_and_log(target_unet, noise_scheduler, CFG, args, accelerator, epoch, "target")
                        
                        if not args.debug:
                            accelerator.log({"avg_loss": avg_loss})
                    
                    # save new checkpoints and clear old checkpoints
                    if epoch % args.checkpoint_interval == 0:
                        # check if this save would set us over the `checkpoints_total_limit`, 
                        if args.checkpoints_total_limit is not None:
                            checkpoints = os.listdir(save_root)
                            checkpoints = [d for d in checkpoints if d.startswith("unet")]
                            checkpoints = sorted(checkpoints, key=lambda x: int(x.split("_")[-1].split(".")[0]))
                            # remove checkpoints exceeding checkpoints_total_limit
                            if len(checkpoints) >= args.checkpoints_total_limit:
                                num_to_remove = len(checkpoints) - args.checkpoints_total_limit + 1
                                removing_checkpoints = checkpoints[0:num_to_remove]
                                logger.info(
                                    f"{len(checkpoints)} checkpoints already exist, removing checkpoints: {', '.join(removing_checkpoints)}"
                                )
                                for removing_checkpoint in removing_checkpoints:
                                    removing_checkpoint = os.path.join(save_root, removing_checkpoint)
                                    os.remove(removing_checkpoint)
                        
                        # save ckpt if loss is better for resume training or later epochs.
                        if epoch > args.epochs * 0.75 or args.resume_pth is not None :
                            if avg_loss < best_loss :
                                best_loss = avg_loss
                                torch.save({
                                    'epoch': epoch,
                                    'model': unet.state_dict(),
                                    'optimizer': optimizer.state_dict(),
                                    'loss': avg_loss
                                    }, 
                                    os.path.join(save_root, f"unet_best_{epoch}.pth")
                                )
                                logger.info(f"unet_best_{epoch}.pth saved to {save_root}")
                                if not args.debug:
                                    accelerator.log({"Ckpt_saved?": 1})
                            else:
                                print(f"epoch {epoch} not saved.")
                                if not args.debug:
                                    accelerator.log({"Ckpt_saved?": 0})
                            
                        # not resume or epoch is < 3/4 of total epochs    
                        else:
                            torch.save({
                                'epoch': epoch,
                                'model': unet.state_dict(),
                                'optimizer': optimizer.state_dict(),
                                "loss" : avg_loss
                                }, 
                                os.path.join(save_root, f"unet_{epoch}.pth")
                            )
                            logger.info(f"unet_{epoch}.pth saved to {save_root}")
                            if not args.debug:    
                                accelerator.log({"Ckpt_saved?": 1})
                            
                        # accelerator.save_state(save_root)
                        # save_state saves all the thing from accelerator.prepare, and save them into directories.
                        #print(f"Saved state to {save_root}")
            

            logs = {"loss": loss.detach().item(), "lr": lr_scheduler.get_last_lr()[0], "epoch": epoch}
            progress_bar.set_postfix(**logs)
            #accelerator.log(logs, step=global_step)
            if not args.debug:
                accelerator.log(logs)
            if global_step >= max_train_steps:
                break
                    
    accelerator.wait_for_everyone()
    if accelerator.is_main_process:
        #unet = accelerator.unwrap_model(unet)
        #unet.save_pretrained(os.path.join(args.save_path, "unet"))
    
        #target_unet = accelerator.unwrap_model(target_unet)
        #target_unet.save_pretrained(os.path.join(args.save_path, "unet_target"))
        torch.save({
            'epoch': epoch,
            'model': target_unet.state_dict(),
            'optimizer': optimizer.state_dict(),
        }, os.path.join(save_root, f"target_unet.pth"))
        
#         if args.push_to_hub:
#             upload_folder(
#                 repo_id=repo_id,
#                 folder_path=args.output_dir,
#                 commit_message="End of training",
#                 ignore_patterns=["step_*", "epoch_*"],
#             )
    
    accelerator.end_training()
    
if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    # ----General----
    parser.add_argument("--debug", action=store_true, help="if debug, wandb logging disabled.")
    parser.add_argument('--exp', type=str, default="CelebA_128_cd", help="experiment directory name")
    parser.add_argument('--save_path', type=str, default="adaptive_linear1", help="output directory")
    parser.add_argument("--seed", type=int, default=None, help="A seed for reproducible training.")
    parser.add_argument("--resume_pth", type=str, default=None, help="such as unet_30.pth")
    # ----Logging----
    parser.add_argument("--logging_dir", type=str, default="logs",
        help=(
            "[TensorBoard](https://www.tensorflow.org/tensorboard) log directory. Will default to"
            " *output_dir/runs/**CURRENT_DATETIME_HOSTNAME***."
        ),)
    # ----Dataset parameters and conditions----
    parser.add_argument('--data', type=str, default=r"data/CelebA_train", help="dataset location")
    parser.add_argument('--pretrained_nums_cond', nargs='+', type=int, default=[4,2], help="number of classes for pretrained model")
    parser.add_argument('--dataset_nums_cond', nargs='+', type=int, default=None, help="number of conditions for dataset, equals to pretrained_nums_cond if not specified.")
    parser.add_argument('--img_size', type=int, default=128, help="training image size")
    parser.add_argument('--pretrained_pth', type=str, default='teacher.pth', help="teacher model name, please put under the same dir as chkpt")
    # ----Batch Size and Training Steps----
    parser.add_argument('--train_batch_size', type=int, default=16, help="training batch size")
    parser.add_argument('--epochs', type=int, default=60, help="total training epochs")
    # ----Network architecture settings----
    parser.add_argument('--arch', type=str, default='unetic', help="unet architecture")
    parser.add_argument('--num_timestep', type=int, default=1000, help="training timesteps of pretrained model")
    parser.add_argument('--emb_size', type=int, default=128, help="embedding's output dimension")
    parser.add_argument('--time_cond_proj_dim', type=int, default=128 ,help="dimension of the guidance scale embedding in the U-Net.")
    parser.add_argument('--num_res_blocks', type=int, default=2, help="number of residual blocks in unet")
    parser.add_argument('--channel_mult', type=list, default=[1, 2, 4, 4], help="width of unet model")
    # ----Optimizer (Adam)----
    parser.add_argument("--adam_beta1", type=float, default=0.9, help="The beta1 parameter for the Adam optimizer.")
    parser.add_argument("--adam_beta2", type=float, default=0.999, help="The beta2 parameter for the Adam optimizer.")
    parser.add_argument("--adam_weight_decay", type=float, default=0.0, help="Weight decay to use.")
    parser.add_argument("--adam_epsilon", type=float, default=1e-08, help="Epsilon value for the Adam optimizer")
    parser.add_argument("--max_grad_norm", default=1.0, type=float, help="Max gradient norm.")
    # ----Learning Rate----
    parser.add_argument('--lr', type=float, default=5e-6, help="learning rate")
    parser.add_argument('--lr_schedule', type=str, default="constant_with_warmup", choices=["linear", "cosine", "cosine_with_restarts", "polynomial", "constant", "constant_with_warmup"],
            help=(
            'The scheduler type to use. Choose between ["linear", "cosine", "cosine_with_restarts", "polynomial",'
            ' "constant", "constant_with_warmup"]'
        ),)
    parser.add_argument("--lr_warmup_steps", type=int, default=500, help="warmup steps for lr scheduler.")
    # ----Checkpoint and Eval----
    parser.add_argument("--checkpoint_interval", type=int, default=2, help="how many epochs per checkpoint saved")
    parser.add_argument("--checkpoints_total_limit", type=int, default=8, help="Max number of checkpoints to store.")
    parser.add_argument('--eval_interval', type=int, default=2, help="Run validation every X epochs")
   # ----Latent Consistency Distillation (LCD) Specific Arguments----
    parser.add_argument("--stepfuse_method", type=str, default="piecewise_linear", choices=["piecewise_linear", "only_ode", "only_teacher", "sigmoid", "exponential"])
    parser.add_argument("--piecewise_segments", type=str, nargs="+", help="Piecewise linear segments in key:value string format.")
    # parser.add_argument("--", type=, default=)
    
    parser.add_argument("--w_interval", nargs=2, type=float, default=[2.6, 3.0], help="The interval of w for guidance scale sampling when training")
    parser.add_argument("--num_ddim_timesteps", type=int, default=50, help="number of timesteps for DDIM sampling.")
    parser.add_argument("--loss_type", type=str, default="huber", choices=["l2", "huber"], help="The type of loss to use for the LCD loss.")
    parser.add_argument("--huber_c", type=float, default=0.001, help="The huber loss parameter. Only used if `--loss_type=huber`.")
    parser.add_argument("--timestep_scaling_factor", type=float, default=10.0, help=(
        "The multiplicative timestep scaling factor used when calculating the boundary scalings for LCM."
        "The higher the scaling is, the lower the approximation error, but the default value of 10.0 should typically suffice."),)
    # ----Exponential Moving Average (EMA)----
    parser.add_argument("--ema_decay", type=float, default=0.95, help="The exponential moving average (EMA) rate or decay factor.") 
    # ----Distributed Training----
    parser.add_argument("--local_rank", type=int, default=-1, help="For distributed training: local_rank")
    # ----CCDM settings, could ignore----
    parser.add_argument('--only_table', action='store_true', help="only use embedding table for class embedding")
    parser.add_argument('--concat', action="store_true", help="concat label embedding before CA")
    parser.add_argument('--projection_dim', type=int, default=512, help='q, k, v dimension in attention layer')
    parser.add_argument('--num_head_channels', type=int, default=-1, help='attention head channels')
    parser.add_argument('--num_heads', type=int, default=-1, help='number of attention heads, either specify head_channels or num_heads')
    parser.add_argument('--ignored', type=str, nargs='+', default=None, help='exclude folder when loading dataset, for compositional zero-shot generation')
    parser.add_argument('--use_spatial_transformer', action="store_true", help="use transfomer based model to do attention")
    parser.add_argument('--compose', action="store_true", help="use compoisition network")

    args = parser.parse_args()
    
    main(args)
    
    