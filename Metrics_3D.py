from medpy import metric
import os
from tqdm import tqdm
import numpy as np
import SimpleITK as sitk
import glob
import csv

def check_directories(pre, gt, nii):
    def check_suffix(directory, suffix):
        for filename in os.listdir(directory):
            file_path = os.path.join(directory, filename)
            if os.path.isfile(file_path) and not filename.endswith(suffix):
                return False
        return True

    def count_files(directory):
        return len([f for f in os.listdir(directory) if os.path.isfile(os.path.join(directory, f))])

    all_correct = True

    for directory in [pre, gt]:
        if not check_suffix(directory, '.nii.gz'):
            raise ValueError(f"Directory {directory} does not contain only files ending with .nii.gz.")

    if not check_suffix(nii, '_0000.nii.gz'):
        raise ValueError(f"Directory {nii} does not contain only files ending with _0000.nii.gz.")

    counts = [count_files(pre), count_files(gt), count_files(nii)]
    if len(set(counts)) > 1:
        raise ValueError("The three directories do not have the same number of files.")
    else:
        print("The three directories have the same number of files.")

    if all_correct:
        print("All checks passed.")
    else:
        raise ValueError("Some checks failed.")

    return all_correct

def calculate_metric_percase(pred, gt, voxelspacing):
    pred[pred > 0] = 1
    gt[gt > 0] = 1
    if pred.sum() > 0 and gt.sum() > 0:
        dice = metric.binary.dc(pred, gt)
        hd95 = metric.binary.hd95(pred, gt, voxelspacing)
        asd = metric.binary.asd(pred, gt, voxelspacing)
        return dice, hd95, asd
    elif pred.sum() > 0 and gt.sum() == 0:
        return 1, 0, 0
    else:
        return 0, 0, 0

def trans_pix(img_fdata):
    img_fdata0 = np.zeros((img_fdata.shape))
    for k in range(img_fdata.shape[-1]):
        for i in range(img_fdata.shape[0]):
            for j in range(img_fdata.shape[1]):
                if img_fdata[i, j, k] == 0:
                    img_fdata0[i, j, k] = 0
                elif img_fdata[i, j, k] == 50:
                    img_fdata0[i, j, k] = 1
                elif img_fdata[i, j, k] == 100:
                    img_fdata0[i, j, k] = 2
                elif img_fdata[i, j, k] == 150:
                    img_fdata0[i, j, k] = 3
                else:
                    img_fdata0[i, j, k] = 0
    return img_fdata0

def Metrics_3D(pre_path, gt_path, nii_path, csv_path):
    data = []
    metric_list_dc = 0.0
    metric_list_hd = 0.0
    metric_list_asd = 0.0
    pre_files = glob.glob(os.path.join(pre_path, '*.nii.gz'))
    for f in tqdm(pre_files, total=len(pre_files)):
        metric_dc = []
        metric_hd = []
        metric_asd = []
        _, gt_name = os.path.split(f)
        img_name = f.replace('.nii.gz', '_0000.nii.gz')
        _, filename = os.path.split(img_name)
        img = sitk.ReadImage(os.path.join(nii_path, filename))
        sapcing = img.GetSpacing()
        pre_name = f
        pre_nii = sitk.ReadImage(os.path.join(pre_path, pre_name))
        pre_nii = sitk.GetArrayFromImage(pre_nii)
        gt_nii = sitk.ReadImage(os.path.join(gt_path, gt_name))
        gt_nii = sitk.GetArrayFromImage(gt_nii)

        metric_dc.append(
            calculate_metric_percase(np.isin(pre_nii, [1, 2, 3]), np.isin(gt_nii, [1, 2, 3]), voxelspacing=sapcing)
        )
        metric_dc.append(
            calculate_metric_percase(np.isin(pre_nii, [2, 3]), np.isin(gt_nii, [2, 3]), voxelspacing=sapcing)
        )
        metric_dc.append(
            calculate_metric_percase(np.isin(pre_nii, [3]), np.isin(gt_nii, [3]), voxelspacing=sapcing)
        )

        mean_dc = metric_dc[0][0] + metric_dc[1][0] + metric_dc[2][0]
        mean_hd = metric_dc[0][1] + metric_dc[1][1] + metric_dc[2][1]
        mean_asd = metric_dc[0][2] + metric_dc[1][2] + metric_dc[2][2]
        tqdm.write(f"ASD: {mean_asd / 3}, Dice: {mean_dc / 3}")
        data.append((f,
                     metric_dc[0][0], metric_dc[1][0], metric_dc[2][0], mean_dc / 3,
                     metric_dc[0][1], metric_dc[1][1], metric_dc[2][1], mean_hd / 3,
                     metric_dc[0][2], metric_dc[1][2], metric_dc[2][2], mean_asd / 3))
        metric_list_dc += np.array(metric_dc)
        metric_list_hd += np.array(metric_hd)
        metric_list_asd += np.array(metric_asd)

    with open(csv_path, 'w', newline='') as t_file:
        csv_writer = csv.writer(t_file)
        csv_writer.writerow(('dir',
                             'whole tumor_dc', 'tumor core_dc', 'enhancing tumor_dc', 'mean_dc',
                             'whole tumor_hd', 'tumor core_hd', 'enhancing tumor_hd', 'mean_hd',
                             'whole tumor_asd', 'tumor core_asd', 'enhancing tumor_asd', 'mean_asd'))
        for l in data:
            csv_writer.writerow(l)

if __name__ == '__main__':
    print('This file is used to calculate 3D evaluation metrics.')
    pre_path = r""
    gt_path = r""
    nii_path = r""
    csv_path = r""
    Metrics_3D(pre_path, gt_path, nii_path, csv_path)
