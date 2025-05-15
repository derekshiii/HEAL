import nibabel as nib
import numpy as np
import os
from tqdm import tqdm
def load_nii(file_path):
    nii = nib.load(file_path)
    return nii.get_fdata(), nii
def apply_uncertainty_map(pred_labels, uncertainty_map):
    certainty_mask = 1 - uncertainty_map
    new_prediction = pred_labels * certainty_mask
    return new_prediction
def save_nii(data, reference_nii, output_path):
    new_nii = nib.Nifti1Image(data.astype(np.float32), reference_nii.affine, reference_nii.header)
    nib.save(new_nii, output_path)

pred_labels_dir = r""

uncertainty_map_dir = r""

output_dir = r""

if not os.path.exists(output_dir):
    os.makedirs(output_dir)

pred_file_names = [f for f in os.listdir(pred_labels_dir) if f.endswith('.nii.gz')]
for pred_file_name in tqdm(pred_file_names):

    pred_labels_path = os.path.join(pred_labels_dir, pred_file_name)

    uncertainty_map_path = os.path.join(uncertainty_map_dir, pred_file_name)

    output_path = os.path.join(output_dir, pred_file_name)

    # Load prediction labels and uncertainty map
    pred_labels, pred_nii = load_nii(pred_labels_path)
    uncertainty_map, _ = load_nii(uncertainty_map_path)

    filtered_prediction = apply_uncertainty_map(pred_labels, uncertainty_map)

    save_nii(filtered_prediction, pred_nii, output_path)

print(f"Filtered predictions saved to {output_dir}")