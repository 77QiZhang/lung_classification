import argparse
import json
import logging
import time
import sys
import cv2
from torchvision import models, transforms

import numpy as np
import torch
import matplotlib.pyplot as plt

from sklearn.metrics import roc_curve, auc

print('CUDA_available', torch.cuda.is_available())
import torch.optim as optim
import google.protobuf
print(google.protobuf.__version__)
import torch.nn as nn
import torch.nn.parallel
import torch.utils.data
import torch.utils.data.distributed
from torch.autograd import Function
from torch.utils.tensorboard import SummaryWriter

from warmup_scheduler import GradualWarmupScheduler

import monai
from monai.data import DataLoader, CacheDataset, Dataset
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
from monai.transforms.utils import compute_divisible_spatial_size
from monai.visualize import plot_2d_or_3d_image
from monai.utils import set_determinism
from monai.data.utils import no_collation

# from visualize_image import (
#     box_points_train,
# )

# from models.convnext import convnext_tiny


#get_img_tensor ----> image转换为tensor
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
        default=3,
        help="Select the GPU you want to use",
    )

    args = parser.parse_args()# 解析命令行参数
    
    set_determinism(seed=0)

    amp = True
    if amp:
        compute_dtype = torch.float16
    else:
        compute_dtype = torch.float32    
        
    monai.config.print_config()#打印monai配置库的信息。 包括当前的环境设置、MONAI 版本信息、PyTorch 版本信息等

    torch.set_num_threads(4)#多线程

    env_dict = json.load(open(args.environment_file, "r"))

    for k, v in env_dict.items():
        setattr(args, k, v)
        
    # 设置全局参数
    batch_size = 20
    num_classes = 2

    
    #数据参数
    spatial_size=(64, 64, 32)
    
    spacing = (0.7, 0.7, 1.5)
    image_key = 'image'
    label_key = 'label'
    mask_key = 'mask'
    img_msk_key = [image_key, mask_key]
    larger_patch_size = [spatial_size[0] + 16, spatial_size[1] + 16, spatial_size[2] + 16]
    file = "/homes/clwang/Data/LIDC-IDRI-Crops-Norm/data-minmax/test_datalist_8-2_minmax_remove3(ver3)_feature_sphericity.json"
    with open(file) as f:
        content = json.load(f)
    json_file = '/homes/qzhang/Data/lungwcl/resnet18_test.json'    
     
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
    
    test_ds = CacheDataset(
            data=content,
            transform=test_transforms,
            cache_rate= 1,
        )       

    test_loader = DataLoader(
            test_ds,
            batch_size = 1,
            shuffle = False,
            num_workers = 1,
            pin_memory = torch.cuda.is_available(),
            collate_fn = no_collation,
            persistent_workers = True,
        )
    
    if torch.cuda.is_available():
        DEVICE = torch.device(f"cuda:{args.Choose_GPU}")
    else:
        DEVICE = torch.device("cpu") 
    
    criterion = nn.BCEWithLogitsLoss()
        
    model_ft = torch.load(env_dict["model_path"])
    model_ft.eval()
    
    val_targets_all = []
    val_outputs_softmax_all = []
    image_list = []
    with torch.no_grad():
        for idx, test_data in enumerate(test_loader):
            input_images = [batch_data_ii["image"].to(DEVICE) for batch_data_i in test_data for batch_data_ii in batch_data_i] 
            input_images = torch.stack(input_images, dim=0)

            # image_path = [batch_data_ii["path"] for batch_data_i in test_data for batch_data_ii in batch_data_i]
            
            #label:List[Tensor])
            label_tensors = [batch_data_ii["label"] for batch_data_i in test_data for batch_data_ii in batch_data_i]
            label_array_values = [np.unique(np.array(label).astype(np.int64)).tolist() for label in label_tensors]
            # Creating one-hot labels
            label_origin = torch.zeros((len(test_data), num_classes), dtype=torch.int)
            for i in range(len(label_array_values)):
                value = label_array_values[i]
                for j in range(len(value)):
                    index = int(value[j])
                    label_origin[i][index] = 1        
            target = label_origin.to(DEVICE)
            
            if amp:
                with torch.cuda.amp.autocast():#显式触发垃圾回收
                    test_outputs = model_ft(input_images)#validation
            else:
                test_outputs = model_ft(input_images)  

            loss = criterion(test_outputs.float(), target.float())
            test_outputs_softmax = torch.softmax(test_outputs, dim=1)    
            
            image_dic ={
                'label':label_array_values,
                "target": target.cpu().detach().numpy().tolist(),
                "test_outputs_softmax": test_outputs_softmax.cpu().detach().numpy().tolist()
            }
            image_list.append(image_dic)
            # save outputs for evaluation
            val_targets_all += [target_i.cpu().detach().numpy()[1] for target_i in target]
            val_outputs_softmax_all += [val_outputs_softmax_i.cpu()[1].detach().numpy() for val_outputs_softmax_i in test_outputs_softmax]        
        # compute metrics
        del input_images
        torch.cuda.empty_cache()   
        
        with open(json_file,'w') as fobj:
            fobj.write(json.dumps(image_list, indent=4))   
        # 假设 val_targets_all 和 val_outputs_all 分别是真实标签和模型的预测概率
        fpr, tpr, thresholds = roc_curve(val_targets_all, val_outputs_softmax_all)  
        
        # 初始化最佳阈值和最大的TPR-FPR值
        best_threshold = None
        max_tpr_minus_fpr = -1
        best_acc = 0
       # 遍历各个阈值
        
        for i in range(len(thresholds)):
            threshold = thresholds[i]
            # 根据当前阈值计算预测结果
            predicted_labels = (val_outputs_softmax_all >= threshold).astype(int)
            TP, FP, TN, FN = 0, 0, 0, 0
            # 计算TP、FP、TN、FN
            for pred, true in zip(predicted_labels, val_targets_all):
                # 计算预测结果
                if pred == 1 and true == 1:
                    TP += 1
                elif pred == 1 and true == 0:
                    FP += 1
                elif pred == 0 and true == 0:
                    TN += 1
                elif pred == 0 and true == 1:
                    FN += 1
            # 计算TPR和FPR
            current_tpr = TP / (TP + FN)
            current_fpr = FP / (FP + TN)
            acc = (TP + TN ) / (TP + FP + TN + FN) 
            # 计算当前阈值下的TPR-FPR值
            current_tpr_minus_fpr = current_tpr - current_fpr
            # 更新最佳阈值和最大的TPR-FPR值
            if current_tpr_minus_fpr > max_tpr_minus_fpr:
                best_threshold = threshold
                max_tpr_minus_fpr = current_tpr_minus_fpr
                best_acc = acc
        print("Best Threshold:", best_threshold) 
        print("max_tpr_minus_fpr", max_tpr_minus_fpr)
        print('best_acc', best_acc)
        # 计算 ROC 曲线下方的面积 AUC
        roc_auc = auc(fpr, tpr)
        print('test_roc_auc', roc_auc)               
        plt.figure()
        plt.plot(fpr, tpr, color='darkorange', lw=2, label='ROC curve (area = %0.2f)' % roc_auc)
        plt.plot([0, 1], [0, 1], color='navy', lw=2, linestyle='--')
        plt.xlabel('False Positive Rate')
        plt.ylabel('True Positive Rate')
        plt.title('Receiver Operating Characteristic (ROC) Curve')
        plt.legend(loc='lower right')
        plt.savefig(env_dict["test_roc_path"])
        plt.show()        
        
if __name__ == "__main__":
    logging.basicConfig(
        stream=sys.stdout,
        level=logging.INFO,
        format="[%(asctime)s.%(msecs)03d][%(levelname)5s](%(name)s) - %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
    )
    main()


