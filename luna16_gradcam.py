import argparse
import os
os.environ['ETS_TOOLKIT'] = 'null'
import json
import numpy as np
import torch
import nibabel as nib
from pathlib import Path
print('CUDA_available', torch.cuda.is_available())
import matplotlib.pyplot as plt
from sklearn.metrics import roc_curve, auc

import google.protobuf
print(google.protobuf.__version__)

import torch.nn.parallel
import torch.utils.data
import torch.utils.data.distributed


import monai
from monai.transforms import (
    Compose,
    LoadImageD,
    EnsureChannelFirstD,
    SpacingD,
    LambdaD,
    RandCropByPosNegLabelD,
    CastToTyped,
    ResizeWithPadOrCropD,
    Compose,
    adaptor,
)
from monai.utils import set_determinism
from monai_ex.visualize import GradCAMEx as GradCAMex, LayerCAM

from monai.visualize import GradCAM

def main():
    #参数设置
    parser = argparse.ArgumentParser(description="PyTorch Object Classification Training")
    parser.add_argument(
        "-e",
        "--environment-file",
        default="./config/luna_environment.json",
        help="environment json file that stores environment path",
    )
    
    parser.add_argument(
        "-GPU",
        "--Choose-GPU",
        default=2,
        help="Select the GPU you want to use",
    )
    args = parser.parse_args()# 解析命令行参数
    
    set_determinism(seed=0)
    
    if torch.cuda.is_available():
        DEVICE = torch.device(f"cuda:{args.Choose_GPU}")
    else:
        DEVICE = torch.device("cpu")   
        
    monai.config.print_config()#打印monai配置库的信息。 包括当前的环境设置、MONAI 版本信息、PyTorch 版本信息等

    torch.set_num_threads(4)#多线程

    env_dict = json.load(open(args.environment_file, "r"))
    model_ft = torch.load(env_dict["model_path"])
    model_ft.eval()
    print(model_ft)
    for k, v in env_dict.items():
        setattr(args, k, v)
    
    amp = True
    if amp:
        compute_dtype = torch.float16
    else:
        compute_dtype = torch.float32   
    #数据参数
    spatial_size=(64, 64, 32)
    save_dir = Path('/homes/qzhang/Data/lungwcl/GranCAMEx')
    spacing = (0.7, 0.7, 1.5)
    image_key = 'image'
    label_key = 'label'
    mask_key = 'mask'
    img_msk_key = [image_key, mask_key]
    larger_patch_size = [spatial_size[0] + 16, spatial_size[1] + 16, spatial_size[2] + 16]
    file = "/homes/clwang/Data/LIDC-IDRI-Crops-Norm/data-minmax/test_datalist_8-2_minmax_remove3(ver3)_feature_sphericity.json"
    with open(file) as f:
        content = json.load(f)
        
    
    target_layers =  [
        "features.transition1.conv", 
        "features.transition2.conv", 
        "features.transition3.conv", 
        "features.denseblock4.denselayer2.layers.conv2",
        "features.denseblock4.denselayer6.layers.conv2",
        "features.denseblock4.denselayer10.layers.conv2",
        "features.denseblock4.denselayer12.layers.conv2",
        "features.denseblock4.denselayer14.layers.conv2",
        "features.denseblock4.denselayer15.layers.conv2",
        "features.denseblock4.denselayer16.layers.conv2"
                      ]
    grad_camex = LayerCAM(model_ft, target_layers[::-1], hierarchical=True) #[::-1]反着来
    
    target_layer =  "features.transition3.conv"
    grad_cam =  GradCAM(model_ft, target_layer)
    
    test_transforms = Compose([
        LoadImageD(keys=img_msk_key, dtype=np.float32),
        EnsureChannelFirstD(keys=img_msk_key),
        SpacingD(keys=img_msk_key, pixdim=spacing, mode=['bilinear', 'nearest']),
        LambdaD(keys=label_key, func=lambda x: float(x>3)),
        RandCropByPosNegLabelD(
            keys=img_msk_key,
            label_key=mask_key,
            spatial_size=larger_patch_size,
            pos=1.0, neg=0, num_samples=1,
            allow_smaller=True,
        ),
        ResizeWithPadOrCropD(keys=img_msk_key, spatial_size=spatial_size, mode="reflect"),
        CastToTyped(keys=img_msk_key, dtype=np.float32),
    ])
    
    TP, FP, TN, FN = 0, 0, 0, 0
    image_list = []
    json_file = '/homes/qzhang/Data/lungwcl/grand.json'
    for idx, image_path in enumerate(content):
        image_name = image_path['image']
        data = test_transforms(image_path)
        data_affine = nib.load(image_path["image"])
        
        input_image = torch.from_numpy(data[0]["image"]).unsqueeze(0).to(DEVICE)#[1,1,64,64,32]
        mask = data[0]["mask"]
        camex_result = grad_camex(input_image, class_idx = 1, img_spatial_size=input_image.cpu().detach().numpy().squeeze(1).shape[1:])#[1,1,64,64,32] 
        cam_result = grad_cam(input_image, class_idx = 1)
        
        #acc
        label = np.unique(np.array(data[0]["label"]).astype(np.int64))#[0]
        if label == 0:
            label_origin =  torch.tensor([[1, 0]])
        elif label == 1:
            label_origin =  torch.tensor([[0, 1]])     
               
        target = label_origin.to(DEVICE)

        if amp:
            with torch.cuda.amp.autocast():#显式触发垃圾回收
                test_outputs = model_ft(input_image) 
        else:
            test_outputs = model_ft(input_image) 
            
        test_outputs_softmax = torch.softmax(test_outputs, dim=1)    
        if test_outputs_softmax[0][1] >= 0.4226:
            predicted_classes = 1
        else:
            predicted_classes = 0
        true_classes = [target_i.cpu().detach().numpy()[1] for target_i in target]
        
        if predicted_classes == 1 and true_classes[0] == 1:
                    TP += 1
                    path_acc = "TP"
        elif predicted_classes == 1 and true_classes[0] == 0:
                    FP += 1
                    path_acc = "FP"
        elif predicted_classes == 0 and true_classes[0] == 0:
                    TN += 1
                    path_acc = "TN"
        elif predicted_classes == 0 and true_classes[0] == 1:
                    FN += 1
                    path_acc ="FN"
        image_dict = {
            "image_name": image_name,
            'target_out':target.cpu().detach().numpy(),
            "label": label.tolist(),
            "target": int(true_classes[0]),
            "test_outputs_softmax": test_outputs_softmax.cpu().detach().numpy(),
            'predicted_classes': [predicted_classes]
        }  
        image_list.append(image_dict)

        
        if len(input_image.cpu().detach().numpy().squeeze(1).shape[1:]) == 3:
            for j, (img_slice, camex_slice, cam_slice) in enumerate(zip(input_image.cpu().detach().numpy().squeeze(1), camex_result, cam_result)):
                camex_file_name = (
                    f"batch{idx}_{j}_camex_1.nii.gz"
                )
              
                if save_dir:
                    nib.save(
                        nib.Nifti1Image(img_slice.squeeze(), affine = data_affine.affine),
                        save_dir /f"{path_acc}"/ f"batch{idx}_{j}_images.nii.gz",
                    )
                if save_dir:
                    nib.save(
                        nib.Nifti1Image(mask[0,:,:,:], affine = data_affine.affine),
                        save_dir / f"{path_acc}"/f"batch{idx}_{j}_mask.nii.gz",
                    )
                output_camex = camex_slice.transpose(1, 2, 3, 0).squeeze()
                output_cam = cam_slice.cpu().detach().numpy().transpose(1, 2, 3, 0).squeeze()
                
                # plt.figure(figsize=(15, 15))
                # plt.subplot(4, 3, 12)
                
                # plt.imshow(img_slice.squeeze()[:,:,16],  cmap="gray")
                # plt.contour(mask[0,:,:,16], levels=[0.5], colors='r', alpha=1, linewidths=3)
                # plt.imshow(output_cam[:,:,16], cmap="jet", alpha=0.35)
                # plt.title("GradCAM")
                # plt.axis("off")
                if save_dir:
                    nib.save(
                        nib.Nifti1Image(output_camex[:,:,:,-1], affine = data_affine.affine),
                        save_dir / f"{path_acc}"/camex_file_name,
                    )
                    
                if save_dir:
                    nib.save(
                        nib.Nifti1Image(output_cam[:,:,:], affine = data_affine.affine),
                        save_dir / f"{path_acc}"/f"batch{idx}_{j}_cam_1.nii.gz",
                    )

    #             for i in range(3, output_camex.shape[-1]):
                    
    #                 plt.subplot(4, 3, i+1)
                    
    #                 plt.imshow(img_slice.squeeze()[:,:,16], cmap="gray")
    #                 plt.contour(mask[0,:,:,16],  levels=[0.5], colors='r', alpha=1, linewidths=3)
    #                 plt.imshow(output_camex[:,:,16,i], cmap="jet",alpha=0.35)
    #                 plt.title(f"GradCAMEX_{i-3}")
    #                 plt.axis("off")
                       
    #             plt.savefig(f"/homes/qzhang/Data/lungwcl/GranCAMEx_PNGwochi/{path_acc}/batch{idx}_{j}_cam_261012141516.png")
    #             plt.close()
    # print(TN, TP, FN, FP)
    # current_tpr = TP / (TP + FN)
    # current_fpr = FP / (FP + TN)
    #         # 计算当前阈值下的TPR-FPR值
    # current_tpr_minus_fpr = current_tpr - current_fpr   
    # print(current_tpr_minus_fpr)    
    # acc = (TP + TN ) / (TP + FP + TN + FN) 
    # print(acc)  
    # for item in image_list: 
    #     item['target_out'] = item['target_out'].tolist()
    #     item['test_outputs_softmax'] = item['test_outputs_softmax'].tolist()
    # with open(json_file,'w') as fobj:
    #     fobj.write(json.dumps(image_list, indent=4))
    # print(image_list)
if __name__ == "__main__":
    main()
