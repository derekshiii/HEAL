# HEAL: Learning-Free Source-Free Unsupervised Domain Adaptation for Cross-Modality Medical Image Segmentation

This repository contains the official PyTorch implementation for **HEAL**, Our paper has been accepted in The 36th British Machine Vision Conference.

## Overview of the HEAL Framework

![Overview of HEAL Framework](images/overview.png)

## Prerequisites

Before you begin, ensure you have the following prerequisites met:
* Python 3.10+
* PyTorch 2.12+
* CUDA-enabled GPU

## Workflow

### Step 1: Med-DDPM and nnU-Net Pre-training

* **nnU-Net Pre-training:** For training your nnU-Net model on the source domain, please refer to the official nnU-Net repository and documentation: (3d_fullres for BraTS and 2d for polyp)
  
    * [MIC-DKFZ/nnUNet](https://github.com/MIC-DKFZ/nnUNet)
* **DDPM Pre-training:** For pre-training the DDPM, please follow the guidelines and use the code provided by [mobaidoctor/med-ddpm](https://github.com/mobaidoctor/med-ddpm):
  
    * ```python
      python med-ddpm/train_brats.py --XX --XX --XX
      ```
    
      Note： The pretrained diffusion model can be accessed form https://drive.google.com/drive/folders/1gr8x1cVIKV_l8jJdgIor9-gqPndfmqbl?usp=drive_link

---

### Step 2: nnU-Net  Inference

**How to Run:** (Remember to save the soft predictions from nnunet)

```bash
python /media/XX/nnUNet/nnunetv2/inference/predict_from_raw_data.py -i /media/XX/nnUNetFrame/nnUNet_raw/Dataset147_BraTS00/imagesTr -o /media/XX//T1ce2T1 -d 148 -c 3d_fullres -p nnUNetResEncUNetPlans  --save_probability
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
1.  Inputs: Denoised pseudo-labels (from Step 3) and organize it.
2.  Execute the DDPM inference script.
    ```bash
    python preprocess_brats_data.py
    python med-ddpm/sample_brats.py -XX -XX -XX
    ```

---

### Step 5: Edge Extraction and Selection (EGS Module)

**How to Run:**
1.  Inputs: DDPM-Generated images.
2.  Execute the edge extraction and sample selection script:
    ```bash
    python Extract_edge_canny.py
    python Sample_selection.py
    ```

---

### Step 6: Size-Aware Fusion (SAF)

**How to Run:**

1.  Inputs: Selected image inference and HD refined pseudo-label.
2.  Execute the fusion script.
    ```bash
    python dynamic_fusion.py --XX --XX --XX
    ```

## ✅ TODO List

-  Upload source code for **HEAL** framework
-  Release pretrained source models (nnUNet and Diffusion)
-  Upload inference and training scripts
-  Write a detailed usage guide

## 🙏 Acknowledgements

We sincerely thank the developers of the following open-source tools and datasets, which have made our research possible:

- [nnUNet](https://github.com/MIC-DKFZ/nnUNet) for the baseline segmentation framework.
- [mobaidoctor/med-ddpm](https://github.com/mobaidoctor/med-ddpm) for diffusion inference.
