import os
import cv2
import numpy as np
import pandas as pd
from tqdm import tqdm

def load_png(file_path):
    image = cv2.imread(file_path, cv2.IMREAD_GRAYSCALE)
    if image is None:
        raise ValueError(f"Could not load image: {file_path}")
    return image

def calculate_edge_metrics(edge1, edge2):
    assert edge1.shape == edge2.shape, "The two images must have the same dimensions!"
    edge1_binary = (edge1 > 0).astype(bool)
    edge2_binary = (edge2 > 0).astype(bool)
    metrics = np.logical_and(edge1_binary, edge2_binary)
    metrics_count = np.sum(metrics)
    metrics_rate = metrics_count / np.sum(edge2_binary) if np.sum(edge2_binary) > 0 else 0.0
    return metrics_count, metrics_rate

def process_edge_directories(edge_dir1, edge_dir2, output_csv):
    results = []
    
    edge_files1 = [f for f in os.listdir(edge_dir1) if f.lower().endswith(('.png', '.PNG'))]
    
    for edge_file in tqdm(edge_files1, desc="Processing edge"):
        edge_file1_path = os.path.join(edge_dir1, edge_file)
        edge_file2_path = os.path.join(edge_dir2, edge_file)
        
        if not os.path.exists(edge_file2_path):
            tqdm.write(f"Warning: File {edge_file} does not exist in the second directory, skipping.")
            continue
        
        try:
            edge_image1 = load_png(edge_file1_path)
            edge_image2 = load_png(edge_file2_path)
            metrics_count, metrics_rate = calculate_edge_metrics(edge_image1, edge_image2)
            results.append({
                "FileName": edge_file,
                "metricsCount": metrics_count,
                "metricsRate": metrics_rate
            })
        
        except Exception as e:
            print(f"Error processing {edge_file}: {str(e)}")
    
    df = pd.DataFrame(results)
    df.to_csv(output_csv, index=False)
    print(f"Processing complete, results saved to {output_csv}")

def batch_process_edge_directories(base_input_dirs, base_conditions_dir, output_base_csv_dir):
    for i, input_dir in enumerate(base_input_dirs, 1):
        output_csv = os.path.join(output_base_csv_dir, f"Edge_metrics_{i}.csv")
        conditions_dir = os.path.join(base_conditions_dir)
        
        tqdm.write(f"Processing directory {i}/{len(base_input_dirs)}: {input_dir}")
        process_edge_directories(input_dir, conditions_dir, output_csv)
        tqdm.write(f"Directory {i} processing complete.")

if __name__ == "__main__":
    base_input_dirs = [
        # Example: "/path/to/input_dir1",
        # "/path/to/input_dir2"
    ]
    base_conditions_dir = '' 
        # Example: "/path/to/conditions_dir_corresponding_to_all_input_dirs"
        # Or this could be a list matching base_input_dirs if conditions vary per input_dir
    output_base_csv_dir = '' 
        # Example: "/path/to/output_csv_files_directory"

    if not output_base_csv_dir: # Basic check if output_base_csv_dir is set
        print("Error: 'output_base_csv_dir' is not set. Please define it.")
    else:
        os.makedirs(output_base_csv_dir, exist_ok=True)
        if not base_input_dirs or not base_conditions_dir:
             print("Warning: 'base_input_dirs' or 'base_conditions_dir' might be empty. Proceeding if 'output_base_csv_dir' is set.")
        batch_process_edge_directories(base_input_dirs, base_conditions_dir, output_base_csv_dir)