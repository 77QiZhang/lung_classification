from typing import Callable, Any
import os

import torch
import torch.nn as nn
import numpy as np

from strix import strix_networks
from strix.models.cnn.nets.vgg import VGG
from strix.models.cnn.layers.anatomical_gate import AnatomicalAttentionGate as AAG
# from strix.models import CLASSIFICATION_ARCHI

# from strix.models.cnn.utils import set_trainable
from monai.networks.blocks.dynunet_block import get_conv_layer
from monai_ex.networks.layers import Conv, Norm, Pool


def conv2_block(
    ch_in: int,
    ch_mid: int,
    ch_out: int,
    pool: bool,
    conv_type: Callable,
    norm_type: Callable,
    pool_type: Callable,
):
    layers = [
        conv_type(ch_in, ch_mid, kernel_size=3, stride=1, padding=1, bias=True),
        norm_type(ch_mid),
        nn.ReLU(inplace=True),
        conv_type(ch_mid, ch_out, kernel_size=3, stride=1, padding=1, bias=True),
        norm_type(ch_out),
        nn.ReLU(inplace=True),
    ]
    if pool:
        layers += [pool_type(kernel_size=2, stride=2)]
    return nn.Sequential(*layers)


class VGG9AAG(nn.Module):
    def __init__(self, dim, in_channels, num_classes, roi_classes, **kwargs):
        super(VGG9AAG, self).__init__()
        bottleneck_size = kwargs.get("bottleneck_size", 2)
        init_weights = kwargs.get("init_weights", True)
        conv_type: Callable = Conv[Conv.CONV, dim]
        norm_type: Callable = Norm[Norm.BATCH, dim]
        pool_type: Callable = Pool[Pool.MAX, dim]
        avgpool_type: Callable = Pool[Pool.ADAPTIVEAVG, dim]
        # aag_type = kwargs.get("aag_type", 1)
        aag_mode = kwargs.get("mode", "cat")
        aag_act = kwargs.get("act", "sigmoid")

        # Conv blocks (BatchNorm + ReLU activation added in each block)
        self.maxpool = pool_type(kernel_size=2, stride=2)
        self.layer1 = conv2_block(
            in_channels, 64, 128, True, conv_type, norm_type, pool_type
        )
        self.layer2 = conv2_block(128, 256, 256, True, conv_type, norm_type, pool_type)
        self.layer3 = conv2_block(256, 512, 512, False, conv_type, norm_type, pool_type)

        roi_chns = [roi_classes, 128, 256]
        self.roi_convs = nn.ModuleList(
            [
                get_conv_layer(
                    dim,
                    chn,
                    roi_chns[i + 1],
                    kernel_size=3,
                    stride=1,
                    act="relu",
                    norm="batch",
                    bias=True,
                    conv_only=False,
                )
                for i, chn in enumerate(roi_chns[:-1])
            ]
        )

        self.aag_layers = nn.ModuleList(
            [AAG(dim, chn, chn, aag_mode, aag_act) for chn in roi_chns[1:]]
        )

        # FC layers
        output_size = (bottleneck_size,) * dim  # (2,2,2) #For OOM issue
        output_size = tuple(map(int, output_size))
        num_ = np.prod(output_size)
        self.avgpool = avgpool_type(output_size)
        self.classifier = nn.Sequential(
            nn.Linear(512 * num_, 256 * num_),
            nn.ReLU(True),
            nn.Dropout(),
            nn.Linear(256 * num_, 128 * num_),
            nn.ReLU(True),
            nn.Dropout(),
            nn.Linear(128 * num_, num_classes),
        )
        if init_weights:
            self._initialize_weights()

    def _initialize_weights(self):
        for m in self.modules():
            if isinstance(m, (nn.Conv3d, nn.Conv2d, nn.Conv1d)):
                nn.init.kaiming_normal_(m.weight, mode="fan_out", nonlinearity="relu")
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)
            elif isinstance(m, (nn.BatchNorm3d, nn.BatchNorm2d, nn.BatchNorm1d)):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.Linear):
                nn.init.normal_(m.weight, 0, 0.01)
                nn.init.constant_(m.bias, 0)

    def forward(self, x):
        inp, roi = x
        x = self.layer1(inp)
        y = self.maxpool(self.roi_convs[0](roi))

        x = self.aag_layers[0](x, y)
        x = self.layer2(x)

        y = self.maxpool(self.roi_convs[1](y))
        x = self.aag_layers[1](x, y)

        x = self.layer3(x)

        x = self.avgpool(x)
        x = torch.flatten(x, 1)
        x = self.classifier(x)

        return x

def vgg9_aag(dim, in_channels, num_classes, roi_classes, **kwargs):
    r"""VGG 9-layer model with AAG (configuration "S") with batch normalization"""

    return VGG9AAG(dim, in_channels, num_classes, roi_classes, **kwargs)


# @strix_networks.register("2D", "classification", "vgg9_aag")
# @strix_networks.register("3D", "classification", "vgg9_aag")
def strix_vggaag(
    spatial_dims: int,
    in_channels: int,
    out_channels: int,
    act: str,
    norm: str,
    n_depth: int,
    n_group: int,
    drop_out: float,
    is_prunable: bool,
    pretrained: bool,
    pretrained_model_path: str,
    **kwargs: Any
):
    inkwargs = {}
    inkwargs["roi_classes"] = int(kwargs.get("roi_classes", 1))

    return vgg9_aag(spatial_dims, in_channels, out_channels, **inkwargs)


@strix_networks.register("2D", "classification", "vgg9_aag2")
@strix_networks.register("3D", "classification", "vgg9_aag2")
def medlp_vggaag_prod2(
    spatial_dims: int,
    in_channels: int,
    out_channels: int,
    act: str,
    norm: str,
    n_depth: int,
    n_group: int,
    drop_out: float,
    is_prunable: bool,
    pretrained: bool,
    pretrained_model_path: str,
    **kwargs: Any
):
    inkwargs = {}
    inkwargs["roi_classes"] = int(kwargs.get("roi_classes", 1))
    inkwargs["mode"] = "mode2"

    return vgg9_aag(spatial_dims, in_channels, out_channels, **inkwargs)

@strix_networks.register("2D", "classification", "vgg9_aag3")
@strix_networks.register("3D", "classification", "vgg9_aag3")
def strix_vggaag3(
    spatial_dims: int,
    in_channels: int,
    out_channels: int,
    act: str,
    norm: str,
    n_depth: int,
    n_group: int,
    drop_out: float,
    is_prunable: bool,
    pretrained: bool,
    pretrained_model_path: str,
    **kwargs: Any
):
    inkwargs = {}
    inkwargs["roi_classes"] = int(kwargs.get("roi_classes", 1))
    inkwargs["mode"] = "sum"

    return vgg9_aag(spatial_dims, in_channels, out_channels, **inkwargs)

@strix_networks.register("2D", "classification", "vgg9_aag_single")
@strix_networks.register("3D", "classification", "vgg9_aag_single")
def strix_vggaag_s(
    spatial_dims: int,
    in_channels: int,
    out_channels: int,
    act: str,
    norm: str,
    n_depth: int,
    n_group: int,
    drop_out: float,
    is_prunable: bool,
    pretrained: bool,
    pretrained_model_path: str,
    **kwargs: Any
):
    inkwargs = {}
    inkwargs["roi_classes"] = int(kwargs.get("roi_classes", 1))
    inkwargs["mode"] = "mode3"

    return vgg9_aag(spatial_dims, in_channels, out_channels, **inkwargs)


@strix_networks.register("2D", "classification", "vgg9_aag4")
@strix_networks.register("3D", "classification", "vgg9_aag4")
def strix_vggaag4(
    spatial_dims: int,
    in_channels: int,
    out_channels: int,
    act: str,
    norm: str,
    n_depth: int,
    n_group: int,
    drop_out: float,
    is_prunable: bool,
    pretrained: bool,
    pretrained_model_path: str,
    **kwargs: Any
):
    inkwargs = {}
    inkwargs["roi_classes"] = int(kwargs.get("roi_classes", 1))
    inkwargs["mode"] = "mode4"

    return vgg9_aag(spatial_dims, in_channels, out_channels, **inkwargs)
