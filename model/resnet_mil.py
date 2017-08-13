import torch
import torch.nn as nn
import torchvision.models as models
from torch.autograd import Variable
import torch.nn.functional as F

class resnet_mil(nn.Module):
    def __init__(self, opt):
        super(resnet_mil, self).__init__()
        import model.resnet as resnet
        resnet = resnet.resnet101()
        resnet.load_state_dict(torch.load('/media/jxgu/d2tb/model/resnet/resnet101.pth'))
        self.conv = torch.nn.Sequential()
        self.conv.add_module("conv1", resnet.conv1)
        self.conv.add_module("bn1", resnet.bn1)
        self.conv.add_module("relu", resnet.relu)
        self.conv.add_module("maxpool", resnet.maxpool)
        self.conv.add_module("layer1", resnet.layer1)
        self.conv.add_module("layer2", resnet.layer2)
        self.conv.add_module("layer3", resnet.layer3)
        self.conv.add_module("layer4", resnet.layer4)
        self.l1 = nn.Sequential(nn.Linear(2048, 1000),
                                nn.ReLU(True),
                                nn.Dropout(0.5))
        self.att_size = 7
        self.pool_mil = nn.MaxPool2d(kernel_size=self.att_size, stride=0)

    def forward(self, img, att_size=14):
        x0 = self.conv(img)
        x = self.pool_mil(x0)
        x = x.squeeze(2).squeeze(2)
        x = self.l1(x)
        x1 = torch.add(torch.mul(x.view(x.size(0), 1000, -1), -1), 1)
        cumprod = torch.cumprod(x1, 2)
        out = torch.max(x, torch.add(torch.mul(cumprod[:, :, -1], -1), 1))
        return out
