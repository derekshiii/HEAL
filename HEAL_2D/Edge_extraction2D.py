import os
import cv2
import numpy as np
from skimage import feature
from tqdm import tqdm
from pathlib import Path

def load_png(file_path):
    image = cv2.imread(file_path, cv2.IMREAD_GRAYSCALE)
    if image is None:
        raise ValueError(f"Could not load image: {file_path}")
    return image

def canny_edge_detection_2d(image, sigma=3.0):
    edges = feature.canny(image, sigma=sigma)
    return edges

def save_edge_map(edge_mask, output_path):
    edge_image = edge_mask.astype(np.uint8) * 255
    cv2.imwrite(output_path, edge_image)

def process_directory(input_dir, output_dir, sigma=3.0):
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)
    
    files = [f for f in os.listdir(input_dir) if f.lower().endswith(('.png', '.PNG'))]
    
    for file_name in tqdm(files, desc=f"Processing files in {input_dir}"):
        file_path = os.path.join(input_dir, file_name)
        
        try:
            image_data = load_png(file_path)
            edge_mask = canny_edge_detection_2d(image_data, sigma=sigma)
            output_path = os.path.join(output_dir, f"edge_{file_name}")
            save_edge_map(edge_mask, output_path)
            
        except Exception as e:
            print(f"Error processing {file_name}: {str(e)}")

def batch_process_directories(base_dirs, output_base_dir, sigma=3.0):
    for i, input_dir in enumerate(base_dirs, 1):
        output_dir = os.path.join(output_base_dir, f"Edge_{i}_PNG")
        print(f"Processing directory {i}/{len(base_dirs)}: {input_dir}")
        process_directory(input_dir, output_dir, sigma=sigma)
        print(f"Directory {i} processing complete.")

if __name__ == "__main__":
    input_dirs = [
        # Example: "/path/to/your/input_directory_1",
        # "/path/to/your/input_directory_2"
    ]
    output_base_directory = "" 
        # Example: "/path/to/your/output_base_directory"
        # Modify sigma value to change the edge detection sensitivity, e.g., sigma=3.0
    
    if not input_dirs or not output_base_directory:
        print("Please define 'input_dirs' and 'output_base_directory' before running.")
    else:
        batch_process_directories(input_dirs, output_base_directory, sigma=3.0)