import nibabel as nib
import numpy as np
from skimage import feature
import os
from tqdm import tqdm
def load_nii(file_path):
    nii = nib.load(file_path)
    return nii.get_fdata(), nii

def canny_edge_detection_3d(image, sigma=1.0):
    edges = np.zeros_like(image, dtype=np.bool_)
    for z in range(image.shape[2]):
        edges[:, :, z] = feature.canny(image[:, :, z], sigma=sigma)
    return edges

def save_edge_map(edge_mask, output_path, reference_nii):
    new_nii = nib.Nifti1Image(edge_mask.astype(np.float32), reference_nii.affine)
    nib.save(new_nii, output_path)

def process_directory(input_dir, output_dir, threshold=0.1, sigma=1.0):
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)
    files = [f for f in os.listdir(input_dir) if f.endswith('.nii.gz')]

    for file_name in tqdm(files):
        file_path = os.path.join(input_dir, file_name)
        image_data, nii_header = load_nii(file_path)
        edge_mask = canny_edge_detection_3d(image_data, sigma=sigma)
        output_path = os.path.join(output_dir, f"Edge_{file_name}")
        save_edge_map(edge_mask, output_path, nii_header)


input_dir = r"" 
output_dir = r""
process_directory(input_dir, output_dir, threshold=0.1, sigma=1.0)
