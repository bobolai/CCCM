# CCCM
## Compositional Conditioning Consistency Model
### Introduction
Compositional Conditional Consistency Model (CCCM) is a fast and flexible generative model for compositional conditional image synthesis. CCCM distills the strong compositional generalization abilities of Compositional Conditional Diffusion Models (CCDM) into a consistency-model framework, enabling high-fidelity image generation in just 2–4 inference steps (10-20x faster than CCDM)—while maintaining zero-shot generation on unseen attribute-object pairs.

This repository provides the full codebase for CCCM, including novel consistency distillation strategies—Step Fuse, Loss Fuse, and Switch—which blend teacher predictions and diffusion-formulated signals to achieve optimal trade-offs between image quality and compositional accuracy.
All experiments are conducted on the CelebA dataset.

