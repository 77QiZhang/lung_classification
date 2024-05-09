import nibabel as nib
import numpy as np
import json
from pathlib import Path
import matplotlib.pyplot as plt

def volume(file_path, true_volumes =[], pred_volumes=[], iou_list = [], yu = 0.950000000001 ):
    for case in Path(file_path).iterdir():
        if "mask" in case.name:
            nii_img = nib.load(case)
            nii_data_mask = nii_img.get_fdata()
            # 计算体积
            voxel_volume = np.prod(nii_img.header.get_zooms())  # 计算体素的体积
            nonzero_voxels = np.count_nonzero(nii_data_mask)  # 计算非零值的数量
            volume = nonzero_voxels * voxel_volume
            true_volumes.append(volume)
            if volume > 40000:
                print(case)
            
            nii_img_camex_path = f'{file_path}/{case.name[:-11]}camex_123_261012141516.nii.gz'
            nii_img_camex = nib.load(nii_img_camex_path)
            nii_data = nii_img_camex.get_fdata()
            nii_data_pre = np.where(nii_data <= yu, 1, 0)
            # 计算体积
            voxel_volume = np.prod(nii_img_camex.header.get_zooms())  # 计算体素的体积
            nonzero_voxels = np.count_nonzero(nii_data_pre)  # 计算非零值的数量
            volume = nonzero_voxels * voxel_volume
            pred_volumes.append(volume)
            
            # 计算交集的非零值数量
            intersection = np.logical_and(nii_data_mask, nii_data_pre)
            intersection_count = np.count_nonzero(intersection)

            # 计算并集的非零值数量
            union = np.logical_or(nii_data_mask, nii_data_pre)
            union_count = np.count_nonzero(union)

            # 计算交并比
            iou = intersection_count / union_count
            iou_list.append(iou)
    return true_volumes, pred_volumes,  iou_list
print("Volume of the 3D NII:", volume)

def integrate_nii(file_path, key):
    integral_value_list = []
    for case in Path(file_path).iterdir():
        if f"{key}" in case.name:
            # nii_img_camex_path = f'{file_path}/{case.name[:-11]}camex_123_261012141516.nii.gz'
            # 加载 NIfTI 图像数据
            img = nib.load(case)
            img_data = img.get_fdata()
            if np.isnan(img_data).any():
                print(f"Warning: NaN values found in image {case}. Skipping...")
                continue  # 跳过包含 NaN 值的图像
            # 计算体素积分
            integral_value = np.sum(img_data)      
            integral_value_list.append(integral_value)

    return integral_value_list

if __name__ == "__main__":
    FN_path = '/homes/qzhang/Data/lungwcl/GranCAMEx/FN'
    FP_path = '/homes/qzhang/Data/lungwcl/GranCAMEx/FP'
    TN_path = '/homes/qzhang/Data/lungwcl/GranCAMEx/TN'
    TP_path = '/homes/qzhang/Data/lungwcl/GranCAMEx/TP'
    

    FN_integral_value_CAM = integrate_nii(FN_path, key="camex_1")
    FP_integral_value_CAM = integrate_nii(FP_path, key="camex_1")
    TN_integral_value_CAM = integrate_nii(TN_path, key="camex_1")
    TP_integral_value_CAM = integrate_nii(TP_path, key="camex_1")
    
    
    plt.figure(figsize=(20, 10))
    
    plt.subplot(2, 4, 1)
    plt.hist(FN_integral_value_CAM, bins=20, color='blue', edgecolor='black')
    plt.xlabel('Value')
    plt.ylabel('Frequency')
    plt.title('Distribution of FN_integral_value_CAMEX1')
    plt.axvline(np.mean(FN_integral_value_CAM), color='red', linestyle='dashed', linewidth=1, label='Average')  # 添加平均值线
    plt.legend()
    
    plt.subplot(2, 4, 2)
    plt.hist(FP_integral_value_CAM, bins=20, color='blue', edgecolor='black')
    plt.xlabel('Value')
    plt.ylabel('Frequency')
    plt.title('Distribution of FP_integral_value_CAMEX1')
    plt.axvline(np.mean(FP_integral_value_CAM), color='red', linestyle='dashed', linewidth=1, label='Average')  # 添加平均值线
    plt.legend()

    plt.subplot(2, 4, 3)
    plt.hist(TN_integral_value_CAM, bins=20, color='blue', edgecolor='black')
    plt.xlabel('Value')
    plt.ylabel('Frequency')
    plt.title('Distribution of TN_integral_value_CAMEX1')
    plt.axvline(np.mean(TN_integral_value_CAM), color='red', linestyle='dashed', linewidth=1, label='Average')  # 添加平均值线
    plt.legend()

    plt.subplot(2, 4, 4)
    plt.hist(TP_integral_value_CAM, bins=20, color='blue', edgecolor='black')
    plt.xlabel('Value')
    plt.ylabel('Frequency')
    plt.title('Distribution of TP_integral_value_CAMEX1')
    plt.axvline(np.mean(TP_integral_value_CAM), color='red', linestyle='dashed', linewidth=1, label='Average')  # 添加平均值线
    plt.legend()
    plt.savefig(f"/homes/qzhang/Data/lungwcl/CAMEX_1.png")
    # FN_volumetr, FN_volumepr, FN_iou = volume(FN_path, true_volumes =[], pred_volumes=[], iou_list = [])
    # FP_volumetr, FP_volumepr, FP_iou = volume(FP_path, true_volumes =[], pred_volumes=[], iou_list = [])
    # TN_volumetr, TN_volumepr, TN_iou = volume(TN_path, true_volumes =[], pred_volumes=[], iou_list = [])
    # TP_volumetr, TP_volumepr, TP_iou = volume(TP_path, true_volumes =[], pred_volumes=[], iou_list = [])
    
    # avg_FN_iou  = sum(FN_iou)/len(FN_iou)
    # avg_FP_iou  = sum(FP_iou)/len(FP_iou)
    # avg_TN_iou  = sum(TN_iou)/len(TN_iou)
    # avg_TP_iou  = sum(TP_iou)/len(TP_iou)
    
    # plt.figure(figsize=(8, 6))
    # plt.scatter(range(len(TN_iou)), TN_iou, label = "TN_volumetr", color='red', alpha=0.5)
    # plt.scatter(range(len(TP_iou)), TP_iou, label = "TP_volumetr", color='black', alpha=0.5)
    # plt.scatter(range(len(FN_iou)), FN_iou, label = "FN_volumetr", color='yellow', alpha=0.5)
    # plt.scatter(range(len(FP_iou)), FP_iou, label = "FP_volumetr", color='blue', alpha=0.5)
    # plt.xlabel('Case')
    # plt.ylabel('IoU')
    # plt.title('Volume Distribution of 3D NII Cases')
    # plt.grid(True)
    # plt.legend()
    # plt.savefig("/homes/qzhang/Data/lungwcl/iou.png")
    # plt.close()
    
    # plt.figure(figsize=(8, 6))
    # plt.scatter(range(len(TN_volumetr)), TN_volumetr, label = "TN_volumetr", color='red', alpha=0.5)
    # plt.scatter(range(len(TP_volumetr)), TP_volumetr, label = "TP_volumetr", color='black', alpha=0.5)
    # plt.scatter(range(len(FN_volumetr)), FN_volumetr, label = "FN_volumetr", color='yellow', alpha=0.8)
    # plt.scatter(range(len(FP_volumetr)), FP_volumetr, label = "FP_volumetr", color='blue', alpha=0.5)

    # plt.xlabel('Case')
    # plt.ylabel('volume')
    # plt.title('Volume Distribution of 3D NII Cases')
    # plt.grid(True)
    # plt.legend()
    # plt.savefig("/homes/qzhang/Data/lungwcl/volume.png")
    # plt.close()