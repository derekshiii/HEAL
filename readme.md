# HEAL: Learning-Free Source-Free Unsupervised Domain Adaptation for Cross-Modality Medical Image Segmentation

This repository contains the official PyTorch implementation for **HEAL**

## Overview of the HEAL Framework

![Overview of HEAL Framework](images/overview.png)

## Prerequisites

Before you begin, ensure you have the following prerequisites met:
* Python 3.10+
* PyTorch 2.12+
* CUDA-enabled GPU

## Workflow

### Step 1: DDPM and nnU-Net Pre-training

**Description:**
This initial step involves pre-training the foundational models: a Denoising Diffusion Probabilistic Model (DDPM) and an nnU-Net model on your source domain data. HEAL itself is learning-free during the adaptation phase, relying on these pre-trained models.

* **nnU-Net Pre-training:** For training your nnU-Net model on the source domain, please refer to the official nnU-Net repository and documentation:
    * [MIC-DKFZ/nnUNet](https://github.com/MIC-DKFZ/nnUNet)
* **DDPM Pre-training:** For pre-training the DDPM, please follow the guidelines and use the codebase provided by:
    * [mobaidoctor/med-ddpm](https://github.com/mobaidoctor/med-ddpm)

---

### Step 2: nnU-Net  Inference

**How to Run:** (Remember to save the soft predictions from nnunet)

```bash
python /media/XX/nnUNet/nnunetv2/inference/predict_from_raw_data.py -i /media/XX/nnUNetFrame/nnUNet_raw/Dataset147_BraTS00/imagesTr -o /media/XX//T1ce2T1 -d 148 -c 3d_fullres -p nnUNetResEncUNetPlans
```

---

### Step 3: Entropy-NIG Denoising (HD Module)

**How to Run:**
1.  Input: Initial **soft** pseudo-labels from Step 2.
2.  Execute the denoising script:
    ```bash
    python GenerateUncertaintyMap_Entropy.py
    python BatchesMultiply_Um.py
    python GenerateUncertaintyMap_Entropy-NIG.py
    ```

---

### Step 4: DDPM Inference with Denoised Pseudo-Labels

**How to Run:**
1.  Inputs: Denoised pseudo-labels (from Step 3)
2.  Execute the DDPM inference script.
    ```bash
    python med-ddpm/sample_brats.py -XX -XX -XX
    ```

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

