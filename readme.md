# HEAL: Learning-Free Source-Free Unsupervised Domain Adaptation for Cross-Modality Medical Image Segmentation

This repository contains the official PyTorch implementation for **HEAL (Hierarchical Uncertainty-Denoised Labeling for Source-Free Unsupervised Domain Adaptation)**, a novel framework for cross-modality medical image segmentation. HEAL achieves effective domain adaptation *without requiring access to source domain data or further training/fine-tuning of the pre-trained source model*. This strategy inherently ensures superior data privacy and exceptional computational efficiency, addressing critical limitations of existing SFUDA methods, particularly in sensitive medical applications.

## Overview of the HEAL Framework

![Overview of HEAL Framework](images/overview.png)
*(Briefly describe what this overview plot illustrates, e.g., "The figure above outlines the key stages of the HEAL pipeline, from initial nnU-Net inference to the final size-aware fusion.")*

## Prerequisites

Before you begin, ensure you have the following prerequisites met:
* Python 3.8+
* PyTorch 1.10+ (or as required by dependencies)
* CUDA-enabled GPU (recommended for reasonable processing times)
* Dependencies listed in `requirements.txt` (you should create this file). Install them using:
    ```bash
    pip install -r requirements.txt
    ```
* Pre-trained models:
    * A DDPM model (refer to Step 1).
    * An nnU-Net model pre-trained on your source modality (refer to Step 1).

## Workflow & Usage

The HEAL framework involves a sequence of steps to adapt a source-trained model to a target modality without using source data. Below is a breakdown of each step, along with guidance on running the associated code.

---

### Step 1: DDPM and nnU-Net Pre-training

**Description:**
This initial step involves pre-training the foundational models: a Denoising Diffusion Probabilistic Model (DDPM) and an nnU-Net model on your source domain data. HEAL itself is learning-free during the adaptation phase, relying on these pre-trained models.

**Instructions & Code:**
* **nnU-Net Pre-training:** For training your nnU-Net model on the source domain, please refer to the official nnU-Net repository and documentation:
    * [MIC-DKFZ/nnUNet](https://github.com/MIC-DKFZ/nnUNet)
* **DDPM Pre-training:** For pre-training the DDPM, please follow the guidelines and use the codebase provided by:
    * [mobaidoctor/med-ddpm](https://github.com/mobaidoctor/med-ddpm)

You will need these pre-trained model checkpoints for the subsequent steps.

---

### Step 2: nnU-Net Inference (Initial Pseudo-Labels)

**Description:**
Generate initial segmentation predictions (pseudo-labels) on the unlabeled target domain images using the nnU-Net model pre-trained on the source domain (from Step 1).

**How to Run:**
1.  Ensure your target domain images are prepared (e.g., in NIfTI format).
2.  Execute the nnU-Net inference script. (You'll need to provide your specific script and arguments here).
    ```bash
    # Example placeholder command:
    python scripts/step2_nnunet_inference.py \
        --input_folder /path/to/target_domain_images \
        --output_folder /path/to/save/initial_nnunet_predictions \
        --nnunet_model_path /path/to/your/pretrained_nnunet_model_folder
    ```
    *Replace placeholder paths and arguments with your actual script and parameters.*
    *Refer to `scripts/step2_nnunet_inference/README.md` (if you create one) for detailed instructions.*

---

### Step 3: Entropy-NIG Denoising (HUD Module)

**Description:**
This step applies our Hierarchical Uncertainty Denoising (HUD) module. It calculates uncertainty maps based on entropy and Normal-Inverse Gaussian (NIG) distributions from the initial pseudo-labels. These uncertainty maps are then used to refine the pseudo-labels, aiming to mitigate error accumulation.

**How to Run:**
1.  Input: Initial pseudo-labels from Step 2.
2.  Execute the denoising script.
    ```bash
    # Example placeholder command:
    python scripts/step3_hud_denoising.py \
        --input_pseudo_labels /path/to/initial_nnunet_predictions \
        --output_denoised_labels /path/to/save/denoised_pseudo_labels
    ```
    *Modify with your script name and necessary arguments (e.g., uncertainty thresholds, NIG parameters).*
    *Details can be found in `scripts/step3_hud_denoising/README.md`.*

---

### Step 4: DDPM Inference with Denoised Pseudo-Labels

**Description:**
The refined, denoised pseudo-labels from Step 3 are used to guide a pre-trained DDPM. This step leverages the generative capabilities of the DDPM to further enhance the quality and consistency of the pseudo-labels.

**How to Run:**
1.  Inputs: Denoised pseudo-labels (from Step 3) and the original target domain images.
2.  Execute the DDPM inference script.
    ```bash
    # Example placeholder command:
    python scripts/step4_ddpm_inference.py \
        --input_images /path/to/target_domain_images \
        --input_denoised_labels /path/to/save/denoised_pseudo_labels \
        --output_ddpm_refined_labels /path/to/save/ddpm_refined_labels \
        --ddpm_checkpoint /path/to/your/pretrained_ddpm_model.pt
    ```
    *Ensure you specify parameters for the DDPM, such as the number of diffusion steps, guidance scale, etc.*
    *Refer to `scripts/step4_ddpm_inference/README.md`.*

---

### Step 5: Edge Extraction and Best Sample Selection (EGS³ Module)

**Description:**
This step incorporates our Edge Guidance and Strong Sample Selection (EGS³) module. It involves extracting edge information from the images/labels and a strategy to select the most reliable or "strong" pseudo-labeled samples based on confidence or stability criteria.

**How to Run:**
1.  Inputs: DDPM-refined labels (from Step 4), potentially original images or uncertainty maps.
2.  Execute the sample selection script.
    ```bash
    # Example placeholder command:
    python scripts/step5_egs_selection.py \
        --input_ddpm_labels /path/to/save/ddpm_refined_labels \
        --output_strong_samples_info /path/to/save/strong_samples_list.txt \
        --input_images /path/to/target_domain_images # Optional, if needed for edge extraction
    ```
    *Specify parameters for edge detection and sample selection criteria.*
    *Details in `scripts/step5_egs_selection/README.md`.*

---

### Step 6: Size-Aware Fusion (SAF Module)

**Description:**
The final step is our Size-Aware Fusion (SAF) module. It dynamically fuses pseudo-labels (possibly from different refinement stages or for different selected samples) based on the size of the segmentation target. This helps to improve segmentation performance, especially for objects of varying scales.

**How to Run:**
1.  Inputs: Selected strong pseudo-labels (from Step 5) and potentially other intermediate segmentation results.
2.  Execute the fusion script.
    ```bash
    # Example placeholder command:
    python scripts/step6_size_aware_fusion.py \
        --input_labels_to_fuse /path/to/labels_for_fusion_or_strong_samples \
        --output_final_segmentations /path/to/save/final_HEAL_segmentations
    ```
    *Specify parameters related to size criteria and fusion strategy.*
    *Refer to `scripts/step6_size_aware_fusion/README.md`.*

---

## 
