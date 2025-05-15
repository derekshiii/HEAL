import os


def rename_nii_gz_files(root_dir):
    """
    将指定目录及其子目录下所有.nii.gz文件重命名
    :param root_dir: 要操作的根目录
    """
    for root, dirs, files in os.walk(root_dir):
        for file in files:
            if file.endswith('.nii.gz'):
                old_file_path = os.path.join(root, file)
                new_filename = file.replace('_0003.nii.gz', '.nii')
                new_file_path = os.path.join(root, new_filename)
                os.rename(old_file_path, new_file_path)
                #print(f"sucessfully rename {old_file_path}")



if __name__ == "__main__":
    target_directory = "/media/dell/ShiYulong_HDD/Training_Brats2020/flair"  # 这里表示当前目录，你可以替换成实际要操作的目录路径
    rename_nii_gz_files(target_directory)