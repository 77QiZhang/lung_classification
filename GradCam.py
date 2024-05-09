import torch
import torchvision
from torchvision.models import resnet50
import cv2
import numpy as np
import torch.nn as nn
from torch.autograd import Function
from torchvision import models, transforms
import torch
import cv2
import numpy as np
import torch.nn as nn

import torch
import cv2
import numpy as np
from torchvision.models import resnet50
from torch.nn import functional as F

def add_zero(input_data, output_size=(16, 36, 960)):
    """
        给输入的矩阵和输出的尺寸，差的地方填零
        在3，73，762的末尾数据上补全0，变成160，240，240
        """
    x, y, z = input_data.shape  # 举例3,73,762
    target_x, target_y, target_z = output_size

    if x == target_x and y == target_y and z == target_z:
        return input_data

    result = None
    if x <= target_x:
        padding = (target_x - x) // 2
        result = np.zeros([target_x, y, z], dtype=input_data.dtype)
        result[padding:padding + x, :, :] = input_data
    if x > target_x:
        padding = (x - target_x) // 2
        result = input_data[padding: padding + target_x, :, :]
        
class GradCAM:
    def __init__(self, model, target_layer, device):
        self.model = model
        self.target_layer = target_layer
        self.gradients = None
        self.activations = None
        self.model.eval()
        self.device = device
        self.target_layer.register_forward_hook(self.save_gradient)

    def save_gradient(self, module, input, output):
        self.gradients =  output[0]
        self.activations = output
        
    def forward(self, x):
        return self.model(x)

    def __call__(self, x, target_class=None, save_path=None):
        output = self.forward(x)#

        if target_class is None:
            target_class = np.argmax(output.cpu().data.numpy())
        
        one_hot = torch.zeros((1, output.size()[-1]), dtype=torch.float32)
        one_hot[0][target_class] = 1
        one_hot = one_hot.to(x.to(self.device))

        self.model.zero_grad()
        output.backward(gradient=one_hot, retain_graph=True)

        gradients = self.gradients.cpu().data.numpy()
        activations = self.activations.cpu().data.numpy()

        return gradients, activations

