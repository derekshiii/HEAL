import os
import numpy as np
import matplotlib.pyplot as plt
from PIL import Image
from tqdm import tqdm
from scipy.ndimage import uniform_filter
import glob

def calculate_local_entropy_2d(data, window_size=3):
    pad = window_size // 2
    data_pad = np.pad(data, pad, mode='edge')
    windows = np.lib.stride_tricks.sliding_window_view(data_pad, (window_size, window_size))
    windows_flat = windows.reshape(-1, window_size**2)
    hist = np.zeros((windows_flat.shape[0], 10), dtype=np.float32)
    bin_edges = np.linspace(0, 1, 11)
    for i in range(windows_flat.shape[0]):
        hist[i] = np.histogram(windows_flat[i], bins=bin_edges)[0]
    hist += 1e-6
    prob = hist / hist.sum(axis=1, keepdims=True)
    ent = -np.sum(prob * np.log2(prob), axis=1)
    return ent.reshape(data.shape)

def process_single_image(initial_path, refined_path, output_dir, threshold=0.5, window_size=3, save_all=True):
    if save_all:
        uncertainty_dir = os.path.join(output_dir, 'uncertainty')
        mask_dir = os.path.join(output_dir, 'reliability_mask')
        processed_dir = os.path.join(output_dir, 'processed')
        for directory in [uncertainty_dir, mask_dir, processed_dir]:
            os.makedirs(directory, exist_ok=True)

    initial_img = np.array(Image.open(initial_path).convert('L'), dtype=np.float32) / 255.0
    refined_img = np.array(Image.open(refined_path).convert('L'), dtype=np.float32) / 255.0
    diff = np.abs(initial_img - refined_img)
    H = calculate_local_entropy_2d(refined_img, window_size)
    omega = 1. / np.clip(0.5 * H + 0.01, 1e-6, None)
    beta = 0.8 * (1 - H) + 0.01
    alpha = np.clip(0.1 * diff + 1e-3, 1.01, None)
    uncertainty = beta / (omega * (alpha - 1))
    uncertainty_norm = (uncertainty - np.min(uncertainty)) / (np.max(uncertainty) - np.min(uncertainty) + 1e-6)
    reliability_mask = (uncertainty_norm <= threshold).astype(np.float32)
    processed_initial = initial_img * reliability_mask

    if save_all:
        base_name = os.path.basename(initial_path)
        file_name, ext = os.path.splitext(base_name)
        plt.imsave(os.path.join(uncertainty_dir, f"{file_name}_uncertainty{ext}"), uncertainty_norm, cmap='jet', vmin=0, vmax=1)
        plt.imsave(os.path.join(mask_dir, f"{file_name}_mask{ext}"), reliability_mask, cmap='gray')
        plt.imsave(os.path.join(processed_dir, f"{file_name}_processed{ext}"), processed_initial, cmap='gray')

    return uncertainty_norm, reliability_mask, processed_initial

def batch_process_images(initial_dir, refined_dir, output_dir, threshold=0.5, window_size=3, file_pattern="*.png"):
    os.makedirs(output_dir, exist_ok=True)
    initial_files = sorted(glob.glob(os.path.join(initial_dir, file_pattern)))
    refined_files = sorted(glob.glob(os.path.join(refined_dir, file_pattern)))
    if len(initial_files) != len(refined_files):
        raise ValueError(f"File number mismatch: {len(initial_files)} vs {len(refined_files)}")
    print(f"{len(initial_files)} image pairs found")
    for initial_path, refined_path in tqdm(zip(initial_files, refined_files), desc="Processing", total=len(initial_files)):
        initial_name = os.path.basename(initial_path)
        refined_name = os.path.basename(refined_path)
        if initial_name != refined_name:
            print(f"Warning: Filename mismatch - {initial_name} vs {refined_name}")
            continue
        try:
            process_single_image(initial_path, refined_path, output_dir, threshold, window_size)
        except Exception as e:
            print(f"Error processing file {initial_name}: {str(e)}")
            continue

def visualize_uncertainty(image_path, uncertainty, reliability_mask, threshold, figsize=(15, 5)):
    original = np.array(Image.open(image_path).convert('L'), dtype=np.float32) / 255.0
    fig, axes = plt.subplots(1, 3, figsize=figsize)
    axes[0].imshow(original, cmap='gray')
    axes[0].set_title('Original')
    axes[0].axis('off')
    uncertainty_map = axes[1].imshow(uncertainty, cmap='jet', vmin=0, vmax=1)
    axes[1].set_title('Uncertainty')
    axes[1].axis('off')
    fig.colorbar(uncertainty_map, ax=axes[1], fraction=0.046, pad=0.04)
    axes[2].imshow(reliability_mask, cmap='gray')
    axes[2].set_title(f'Reliable Area (th={threshold})')
    axes[2].axis('off')
    plt.tight_layout()
    plt.show()

if __name__ == "__main__":
    initial_dir = ''
    refined_dir = ''
    output_dir = ''
    batch_process_images(
        initial_dir=initial_dir,
        refined_dir=refined_dir,
        output_dir=output_dir,
        threshold=0.2,
        window_size=3,
        file_pattern="*.png"
    )
