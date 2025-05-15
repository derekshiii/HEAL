import os
import nibabel as nib
import numpy as np
import pandas as pd
from tqdm import tqdm

def load_nii(file_path):
    nii = nib.load(file_path)
    return nii.get_fdata()

def calculate_edge_metrics(edge1, edge2):
    assert edge1.shape == edge2.shape, "Input images must have the same shape."
    metrics = np.logical_and(edge1, edge2)
    metrics_count = np.sum(metrics)
    metrics_rate = metrics_count / np.sum(edge2) if np.sum(edge2) > 0 else 0.0
    return metrics_count, metrics_rate

def process_edge_directories(edge_dir1, edge_dir2, output_csv):
    results = []
    edge_files1 = [f for f in os.listdir(edge_dir1) if f.endswith('.nii.gz')]
    for edge_file in tqdm(edge_files1):
        edge_file1_path = os.path.join(edge_dir1, edge_file)
        edge_file2_path = os.path.join(edge_dir2, edge_file)
        if not os.path.exists(edge_file2_path):
            print(f"Warning: {edge_file} not found in second directory. Skipping.")
            continue
        edge_image1 = load_nii(edge_file1_path).astype(bool)
        edge_image2 = load_nii(edge_file2_path).astype(bool)
        metrics_count, metrics_rate = calculate_edge_metrics(edge_image1, edge_image2)
        results.append({
            "FileName": edge_file,
            "metricsCount": metrics_count,
            "metricsRate": metrics_rate
        })
    df = pd.DataFrame(results)
    df.to_csv(output_csv, index=False)
    print(f"Processing completed. Results saved to {output_csv}")

edge_dir1 = ""
edge_dir2 = ""
output_csv = ""
process_edge_directories(edge_dir1, edge_dir2, output_csv)
