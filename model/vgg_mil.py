import torch
import torch.nn as nn
import os
from torch.autograd import Variable
from torch.utils.serialization import load_lua
import torch.nn.functional as F

class vgg_mil(nn.Module):
    def __init__(self, opt):
        super(vgg_mil, self).__init__()
        self.conv = torch.nn.Sequential()
        self.conv.add_module("conv1_1", nn.Conv2d(3, 64, kernel_size=3, stride=1, padding=1))
        self.conv.add_module("relu_1_1", torch.nn.ReLU())
        self.conv.add_module("conv1_2", nn.Conv2d(64, 64, kernel_size=3, stride=1, padding=1))
        self.conv.add_module("relu_1_2", torch.nn.ReLU())
        self.conv.add_module("maxpool_1", torch.nn.MaxPool2d(kernel_size=2))

        self.conv.add_module("conv2_1", nn.Conv2d(64, 128, kernel_size=3, stride=1, padding=1))
        self.conv.add_module("relu_2_1", torch.nn.ReLU())
        self.conv.add_module("conv2_2", nn.Conv2d(128, 128, kernel_size=3, stride=1, padding=1))
        self.conv.add_module("relu_2_2", torch.nn.ReLU())
        self.conv.add_module("maxpool_2", torch.nn.MaxPool2d(kernel_size=2))

        self.conv.add_module("conv3_1", nn.Conv2d(128, 256, kernel_size=3, stride=1, padding=1))
        self.conv.add_module("relu_3_1", torch.nn.ReLU())
        self.conv.add_module("conv3_2", nn.Conv2d(256, 256, kernel_size=3, stride=1, padding=1))
        self.conv.add_module("relu_3_2", torch.nn.ReLU())
        self.conv.add_module("conv3_3", nn.Conv2d(256, 256, kernel_size=3, stride=1, padding=1))
        self.conv.add_module("relu_3_3", torch.nn.ReLU())
        self.conv.add_module("maxpool_3", torch.nn.MaxPool2d(kernel_size=2))

        self.conv.add_module("conv4_1", nn.Conv2d(256, 512, kernel_size=3, stride=1, padding=1))
        self.conv.add_module("relu_4_1", torch.nn.ReLU())
        self.conv.add_module("conv4_2", nn.Conv2d(512, 512, kernel_size=3, stride=1, padding=1))
        self.conv.add_module("relu_4_2", torch.nn.ReLU())
        self.conv.add_module("conv4_3", nn.Conv2d(512, 512, kernel_size=3, stride=1, padding=1))
        self.conv.add_module("relu_4_3", torch.nn.ReLU())
        self.conv.add_module("maxpool_4", torch.nn.MaxPool2d(kernel_size=2))

        self.conv.add_module("conv5_1", nn.Conv2d(512, 512, kernel_size=3, stride=1, padding=1))
        self.conv.add_module("relu_5_1", torch.nn.ReLU())
        self.conv.add_module("conv5_2", nn.Conv2d(512, 512, kernel_size=3, stride=1, padding=1))
        self.conv.add_module("relu_5_2", torch.nn.ReLU())
        self.conv.add_module("conv5_3", nn.Conv2d(512, 512, kernel_size=3, stride=1, padding=1))
        self.conv.add_module("relu_5_3", torch.nn.ReLU())
        self.conv.add_module("maxpool_5", torch.nn.MaxPool2d(kernel_size=2))

        self.conv.add_module("fc6_conv", nn.Conv2d(512, 4096, kernel_size=7, stride=1, padding=0))
        self.conv.add_module("relu_6_1", torch.nn.ReLU())

        self.conv.add_module("fc7_conv", nn.Conv2d(4096, 4096, kernel_size=1, stride=1, padding=0))
        self.conv.add_module("relu_7_1", torch.nn.ReLU())

        self.conv.add_module("fc8_conv", nn.Conv2d(4096, 1000, kernel_size=1, stride=1, padding=0))
        self.conv.add_module("sigmoid_8", torch.nn.Sigmoid())

        self.pool_mil = nn.MaxPool2d(kernel_size=11, stride=0)

        self.weight_init()

    def weight_init(self):
        self.cnn_weight = 'model/vgg16_full_conv_mil.pth'
        self.conv.load_state_dict(torch.load(self.cnn_weight))
        print("Load pretrained CNN model from " + self.cnn_weight)

    def forward(self, x):
        x0 = self.conv.forward(x.float())
        x = self.pool_mil(x0)
        x = x.squeeze(2).squeeze(2)
        x1 = torch.add(torch.mul(x0.view(x.size(0), 1000, -1), -1), 1)
        cumprod = torch.cumprod(x1, 2)
        out = torch.max(x, torch.add(torch.mul(cumprod[:, :, -1], -1), 1))
        out = F.softmax(out)
        return out

class MIL_Precision_Score_Mapping(nn.Module):
    def __init__(self):
        super(MIL_Precision_Score_Mapping, self).__init__()
        self.mil = nn.MaxPool2d(kernel_size=11, stride=0)

    def forward(self, x, score, precision, mil_prob):
        out = self.mil(x)
        return out

