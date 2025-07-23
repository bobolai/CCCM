# CCCM
## Compositional Conditioning Consistency Model
### Introduction
Compositional Conditional Consistency Model (CCCM) is a fast and flexible generative model for compositional conditional image synthesis. CCCM distills the strong compositional generalization abilities of Compositional Conditional Diffusion Models (CCDM) into a consistency-model framework, enabling high-fidelity image generation in just 2–4 inference steps (10-20x faster than CCDM)—while maintaining zero-shot generation on unseen attribute-object pairs.

<img width="627" height="297" alt="image" src="https://github.com/user-attachments/assets/d3d38739-d960-4f0c-ad3c-f5b89ba88bd1" />

This repository provides the full codebase for CCCM, including novel consistency distillation strategies—Step Fuse, Loss Fuse, and Switch—which blend teacher predictions and diffusion-formulated signals to achieve optimal trade-offs between image quality and compositional accuracy.
All experiments are conducted on the CelebA dataset.

### Method Overview
Teacher Model: pretrained CCDM (U-Net), trained for compositional conditional generation.

Student Model: CCCM, initialized from CCDM weights, but trained with consistency loss as proposed in Consistency Models (Song et al., 2023).

Modified Consistency Distillation:
Leverages teacher predictions (via ODE/DDIM solver) and diffusion-formulated supervision.
Three supervision fusion strategies: StepFuse, LossFuse, Switch, each with scheduler-controlled weighting.

<img width="636" height="274" alt="switch_ppt" src="https://github.com/user-attachments/assets/2a93e821-2537-4d84-9578-2f4e6fdfe104" />
<img width="636" height="274" alt="stepfuse_ppt" src="https://github.com/user-attachments/assets/77847c5a-f256-4e5f-b977-d7ed1c02786e" />
<img width="636" height="274" alt="lossfuse_ppt" src="https://github.com/user-attachments/assets/21ed0653-27e8-4d6b-81fe-1f9032951c6a" />

### Installation


