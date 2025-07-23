# CCCM
## Compositional Conditioning Consistency Model
### Introduction
Compositional Conditional Consistency Model (CCCM) is a fast and flexible generative model for compositional conditional image synthesis. CCCM distills the strong compositional generalization abilities of Compositional Conditional Diffusion Models (CCDM) into a consistency-model framework, enabling high-fidelity image generation in just 2–4 inference steps (10-20x faster than CCDM)—while maintaining zero-shot generation on unseen attribute-object pairs.

<img width="488" height="408" alt="consistency_function2" src="https://github.com/user-attachments/assets/2c8e7b19-bb01-42a3-8174-6cc49c647bbc" />


This repository provides the full codebase for CCCM, including novel consistency distillation strategies—Step Fuse, Loss Fuse, and Switch—which blend teacher predictions and diffusion-formulated signals to achieve optimal trade-offs between image quality and compositional accuracy.
All experiments are conducted on the CelebA dataset.

### Method Overview
Teacher Model: pretrained CCDM (U-Net), trained for compositional conditional generation.

Student Model: CCCM, initialized from CCDM weights, but trained with consistency loss as proposed in Consistency Models (Song et al., 2023).

Modified Consistency Distillation:
Leverages teacher predictions (via ODE/DDIM solver) and diffusion-formulated supervision.
Three supervision fusion strategies: StepFuse, LossFuse, Switch, each with scheduler-controlled weighting.

