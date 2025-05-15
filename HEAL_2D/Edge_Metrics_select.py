#!/usr/bin/env python
# -*- coding:utf-8 -*-
import pandas as pd
import os
import shutil
from collections import defaultdict
from tqdm import tqdm

def select_best_edge_metrics_files(csv_files, source_dirs, target_dir):
    os.makedirs(target_dir, exist_ok=True)
    best_edge_metricss = defaultdict(lambda: {'rate': -1, 'source_dir': '', 'filename': ''})

    for i, csv_file in enumerate(csv_files):
        df = pd.read_csv(csv_file)
        for _, row in df.iterrows():
            csv_filename = row['FileName']
            edge_metrics_rate = float(row['Edge_Metrics'])
            file_id = csv_filename.replace("edge_", "")
            if edge_metrics_rate > best_edge_metricss[file_id]['rate']:
                best_edge_metricss[file_id] = {
                    'rate': edge_metrics_rate,
                    'source_dir': source_dirs[i],
                    'filename': file_id
                }
    
    for file_id, info in tqdm(best_edge_metricss.items(), desc="Copying best edge_metrics files"):
        source_path = os.path.join(info['source_dir'], info['filename'])
        target_path = os.path.join(target_dir, file_id)
        try:
            shutil.copy2(source_path, target_path)
            tqdm.write(f"Copied {source_path} to {target_path}, Edge_Metrics: {info['rate']}")
        except Exception as e:
            print(f"Error copying {source_path}: {e}")
            try:
                print(f"Files in source directory {info['source_dir']}:")
                for f_name in os.listdir(info['source_dir']):
                    if file_id in f_name: 
                        print(f" - {f_name} (potential match)")
                    else:
                        print(f" - {f_name}")
            except Exception as list_err:
                print(f"Could not list directory {info['source_dir']}: {list_err}")

if __name__ == "__main__":
    csv_files = [
        # Example: "/path/to/your/data1.csv",
        # "/path/to/your/data2.csv"
    ]
    source_dirs = [
        # Example: "/path/to/source_images_for_data1_csv",
        # "/path/to/source_images_for_data2_csv"
    ]
    target_dir = "" # Example: "/path/to/your/output_selected_files"
    
    # Basic check to ensure paths are set (you might want to use argparse for real use)
    if not csv_files or not source_dirs or not target_dir:
        print("Please set 'csv_files', 'source_dirs', and 'target_dir' variables before running.")
    elif len(csv_files) != len(source_dirs):
        print("Error: The number of CSV files must match the number of source directories.")
    else:
        select_best_edge_metrics_files(csv_files, source_dirs, target_dir)