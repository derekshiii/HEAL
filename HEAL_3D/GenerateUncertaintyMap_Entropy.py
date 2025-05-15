import numpy as np
import nibabel as nib
import os
import time
from tqdm import tqdm

def load_nnunet_probabilities(npz_path, memmap=True):
    if memmap:
        data = np.load(npz_path, mmap_mode='r')
    else:
        data = np.load(npz_path)
    probabilities = data['probabilities'].transpose(1, 2, 3, 0)
    return probabilities

def compute_uncertainty_entropy(probabilities):
    probabilities = np.clip(probabilities, 1e-7, 1 - 1e-7)
    uncertainty = -np.sum(probabilities * np.log(probabilities), axis=-1)
    return uncertainty

def save_uncertainty_map(uncertainty_map, output_path, reference_nii_path, threshold=None):
    reference_nii = nib.load(reference_nii_path)
    reference_shape = reference_nii.get_fdata().shape
    if uncertainty_map.shape != reference_shape:
        raise ValueError(f"Mismatched dimensions: Uncertainty map {uncertainty_map.shape} does not match reference {reference_shape}.")
    if threshold is not None:
        uncertainty_map = (uncertainty_map > threshold).astype(np.uint8)
    uncertainty_nii = nib.Nifti1Image(uncertainty_map, reference_nii.affine)
    nib.save(uncertainty_nii, output_path)

def process_batch(input_dir, output_dir, threshold=None, use_memmap=True):
    os.makedirs(output_dir, exist_ok=True)
    files = [f for f in os.listdir(input_dir) if f.endswith('.npz')]
    print(f"Found {len(files)} .npz files in {input_dir}.")
    for npz_file in tqdm(files, desc="Processing files"):
        base_name = os.path.splitext(npz_file)[0]
        npz_path = os.path.join(input_dir, npz_file)
        reference_nii_path = os.path.join(input_dir, f"{base_name}.nii.gz")
        output_path = os.path.join(output_dir, f"{base_name}.nii.gz")
        if not os.path.exists(reference_nii_path):
            print(f"Warning: Reference file not found for {npz_file}. Skipping.")
            continue
        try:
            probabilities = load_nnunet_probabilities(npz_path, memmap=use_memmap)
            reference_nii = nib.load(reference_nii_path)
            reference_shape = reference_nii.get_fdata().shape
            if probabilities.shape[:3] != reference_shape:
                probabilities = np.transpose(probabilities, (2, 1, 0, 3))
            uncertainty_map = compute_uncertainty_entropy(probabilities)
            save_uncertainty_map(uncertainty_map, output_path, reference_nii_path, threshold=threshold)
        except Exception as e:
            print(f"Error processing {npz_file}: {e}")

if __name__ == "__main__":
    input_dir = ""
    output_dir = ""
    threshold = 0.2
    process_batch(input_dir, output_dir, threshold=threshold)
