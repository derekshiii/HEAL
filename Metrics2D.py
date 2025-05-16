import os
import numpy as np
import pandas as pd
from PIL import Image
from scipy.ndimage import distance_transform_edt
from skimage.measure import label, regionprops
import glob
import csv
import argparse
from tqdm import tqdm

def dice_coefficient(y_true, y_pred):
    y_true = y_true.astype(bool)
    y_pred = y_pred.astype(bool)
    intersection = np.logical_and(y_true, y_pred).sum()
    return (2.0 * intersection) / (y_true.sum() + y_pred.sum() + 1e-10)

def hausdorff_distance_95(y_true, y_pred, voxel_spacing=(1,1)):
    y_true = y_true.astype(bool)
    y_pred = y_pred.astype(bool)
    if not np.any(y_true) or not np.any(y_pred):
        return np.nan
    dist_map_true = distance_transform_edt(~y_true, sampling=voxel_spacing)
    dist_map_pred = distance_transform_edt(~y_pred, sampling=voxel_spacing)
    surface_true = np.logical_and(y_true, ~binary_erosion(y_true))
    surface_pred = np.logical_and(y_pred, ~binary_erosion(y_pred))
    hd1 = dist_map_true[surface_pred]
    hd2 = dist_map_pred[surface_true]
    if len(hd1) == 0 or len(hd2) == 0:
        return np.nan
    hd1_95 = np.percentile(hd1, 95)
    hd2_95 = np.percentile(hd2, 95)
    return max(hd1_95, hd2_95)

def binary_erosion(binary_img, kernel_size=3):
    from scipy.ndimage import binary_erosion as scipy_erosion
    return scipy_erosion(binary_img, structure=np.ones((kernel_size, kernel_size)))

def average_surface_distance(y_true, y_pred, voxel_spacing=(1,1)):
    y_true = y_true.astype(bool)
    y_pred = y_pred.astype(bool)
    if not np.any(y_true) or not np.any(y_pred):
        return np.nan
    dist_map_true = distance_transform_edt(~y_true, sampling=voxel_spacing)
    dist_map_pred = distance_transform_edt(~y_pred, sampling=voxel_spacing)
    surface_true = np.logical_and(y_true, ~binary_erosion(y_true))
    surface_pred = np.logical_and(y_pred, ~binary_erosion(y_pred))
    sd1 = dist_map_true[surface_pred].mean() if np.any(surface_pred) else np.nan
    sd2 = dist_map_pred[surface_true].mean() if np.any(surface_true) else np.nan
    if np.isnan(sd1) or np.isnan(sd2):
        return np.nan
    return (sd1 + sd2) / 3.0

def check_directory_matching(true_dir, pred_dir):
    true_files = sorted(glob.glob(os.path.join(true_dir, "*.png")))
    pred_files = sorted(glob.glob(os.path.join(pred_dir, "*.png")))
    if len(true_files) != len(pred_files):
        print(f"Warning: File count mismatch. GT: {len(true_files)}, Pred: {len(pred_files)}")
        return False, []
    true_basenames = [os.path.basename(f) for f in true_files]
    pred_basenames = [os.path.basename(f) for f in pred_files]
    if set(true_basenames) != set(pred_basenames):
        print("Warning: Filename mismatch.")
        missing_files = set(true_basenames) - set(pred_basenames)
        extra_files = set(pred_basenames) - set(true_basenames)
        if missing_files:
            print(f"Missing in prediction: {missing_files}")
        if extra_files:
            print(f"Extra in prediction: {extra_files}")
        common_files = []
        for true_file in true_files:
            basename = os.path.basename(true_file)
            if basename in pred_basenames:
                pred_idx = pred_basenames.index(basename)
                common_files.append((true_file, pred_files[pred_idx]))
        print(f"Processing {len(common_files)} common files only.")
        return True, common_files
    matching_files = []
    for i, true_file in enumerate(true_files):
        basename = os.path.basename(true_file)
        pred_idx = pred_basenames.index(basename)
        matching_files.append((true_file, pred_files[pred_idx]))
    print(f"Found {len(matching_files)} matched files.")
    return True, matching_files

def check_image_compatibility(true_path, pred_path):
    try:
        true_img = np.array(Image.open(true_path).convert('L'))
        pred_img = np.array(Image.open(pred_path).convert('L'))
        if true_img.shape != pred_img.shape:
            print(f"Warning: Shape mismatch - {os.path.basename(true_path)}")
            print(f"  GT shape: {true_img.shape}, Pred shape: {pred_img.shape}")
            return False, None, None
        true_img = (true_img > 0).astype(np.uint8)
        pred_img = (pred_img > 0).astype(np.uint8)
        return True, true_img, pred_img
    except Exception as e:
        print(f"Error processing image {os.path.basename(true_path)}: {str(e)}")
        return False, None, None

def evaluate_segmentation(true_dir, pred_dir, output_file):
    is_valid, file_pairs = check_directory_matching(true_dir, pred_dir)
    if not is_valid or not file_pairs:
        print("Aborted: directory mismatch or no common files.")
        return
    results = []
    for true_path, pred_path in tqdm(file_pairs, desc="Evaluating"):
        filename = os.path.basename(true_path)
        is_compatible, true_img, pred_img = check_image_compatibility(true_path, pred_path)
        if not is_compatible:
            results.append({
                'filename': filename,
                'dice': np.nan,
                'hd95': np.nan,
                'asd': np.nan,
                'status': 'error: incompatible image'
            })
            continue
        dice = dice_coefficient(true_img, pred_img)
        try:
            hd95 = hausdorff_distance_95(true_img, pred_img)
            asd = average_surface_distance(true_img, pred_img)
            status = 'success'
        except Exception as e:
            print(f"Error computing distances for {filename}: {str(e)}")
            hd95 = np.nan
            asd = np.nan
            status = f'error: {str(e)}'
        results.append({
            'filename': filename,
            'dice': dice,
            'hd95': hd95,
            'asd': asd,
            'status': status
        })
    df = pd.DataFrame(results)
    mean_dice = df['dice'].mean()
    mean_hd95 = df['hd95'].mean()
    mean_asd = df['asd'].mean()
    summary = pd.DataFrame([{
        'filename': 'AVERAGE',
        'dice': mean_dice,
        'hd95': mean_hd95,
        'asd': mean_asd,
        'status': 'summary'
    }])
    df = pd.concat([df, summary])
    df.to_csv(output_file, index=False)
    print(f"Saved results to {output_file}")
    print("\nSummary:")
    print(f"Mean Dice: {mean_dice:.4f}")
    print(f"Mean HD95: {mean_hd95:.4f}")
    print(f"Mean ASD: {mean_asd:.4f}")
    print(f"Successful evaluations: {df['status'].value_counts().get('success', 0)}/{len(file_pairs)}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Evaluate medical image segmentation metrics')
    parser.add_argument('--true_dir', type=str, help='Directory of ground truth images', default=r"")
    parser.add_argument('--pred_dir', type=str, help='Directory of predicted images', default=r"J")
    parser.add_argument('--output', type=str, default=r"", help='Output CSV file')
    args = parser.parse_args()

    evaluate_segmentation(
        r"", 
        r"", 
        r""
    )
