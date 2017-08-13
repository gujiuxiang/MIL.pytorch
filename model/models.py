from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import collections
import torch
import torch.nn as nn
from torch.autograd import Variable
import time
import Image
import os
import os.path as osp
import sys
import platform
import cPickle
import urllib
import cv2, numpy as np
from scipy.interpolate import interp1d
from matplotlib.pyplot import show
import matplotlib.pyplot as plt
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
import torch.backends.cudnn as cudnn
import tensorflow as tf
import torchvision
import torchvision.transforms as transforms
from six.moves import cPickle
from model.vgg_mil import *
from model.resnet_mil import *
from model.utils import *
import itertools

class Criterion(nn.Module):
    def __init__(self):
        super(Criterion, self).__init__()
        #self.loss = nn.MultiLabelMarginLoss()
        self.loss = nn.MultiLabelSoftMarginLoss()
        #self.loss = nn.MultiMarginLoss()
        #self.loss = nn.CrossEntropyLoss()
        #self.loss = nn.NLLLoss()

    def forward(self, input, target):
        output = self.loss(input, target.float())
        return output

def build_mil(opt):
    opt.n_gpus = getattr(opt, 'n_gpus', 1)

    if 'resnet101' in opt.model:
        mil_model = resnet_mil(opt)
    else:
        mil_model = vgg_mil(opt)

    if opt.n_gpus>1:
        print('Construct multi-gpu model ...')
        model = nn.DataParallel(mil_model, device_ids=opt.gpus, dim=0)
    else:
        model = mil_model
    # check compatibility if training is continued from previously saved model
    if len(opt.start_from) != 0:
        # check if all necessary files exist
        assert os.path.isdir(opt.start_from), " %s must be a a path" % opt.start_from
        lm_info_path = os.path.join(opt.start_from, os.path.basename(opt.start_from) + '.infos-best.pkl')
        lm_pth_path = os.path.join(opt.start_from, os.path.basename(opt.start_from) + '.model-best.pth')
        assert os.path.isfile(lm_info_path), "infos.pkl file does not exist in path %s" % opt.start_from
        model.load_state_dict(torch.load(lm_pth_path))
    model.cuda()
    model.train()  # Assure in training mode
    return model

def build_optimizer(opt, model, infos):
    opt.pre_ft = getattr(opt, 'pre_ft', 1)

    #model_parameters = itertools.ifilter(lambda p: p.requires_grad, model.parameters())
    optimize = opt.optim
    if optimize == 'adam':
        optimizer = torch.optim.Adam(model.parameters(), lr=opt.learning_rate, weight_decay=0.0005)
    elif optimize == 'sgd':
        optimizer = torch.optim.SGD(model.parameters(), lr=opt.learning_rate, momentum=0.999, weight_decay=0.0005)
    elif optimize == 'Adadelta':
        optimizer = torch.optim.Adadelta(model.parameters(), lr=opt.learning_rate, weight_decay=0.0005)
    elif optimize == 'Adagrad':
        optimizer = torch.optim.Adagrad(model.parameters(), lr=opt.learning_rate, weight_decay=0.0005)
    elif optimize == 'Adamax':
        optimizer = torch.optim.Adamax(model.parameters(), lr=opt.learning_rate, weight_decay=0.0005)
    elif optimize == 'ASGD':
        optimizer = torch.optim.ASGD(model.parameters(), lr=opt.learning_rate, weight_decay=0.0005)
    elif optimize == 'LBFGS':
        optimizer = torch.optim.LBFGS(model.parameters(), lr=opt.learning_rate, weight_decay=0.0005)
    elif optimize == 'RMSprop':
        optimizer = torch.optim.RMSprop(model.parameters(), lr=opt.learning_rate, weight_decay=0.0005)

    infos['optimized'] = True

    # Load the optimizer
    if len(opt.start_from) != 0:
        if os.path.isfile(os.path.join(opt.start_from, opt.model_id + '.optimizer.pth')):
            optimizer.load_state_dict(torch.load(os.path.join(opt.start_from, opt.model_id + '.optimizer.pth')))

    return optimizer, infos

def build_models(opt, infos):
    model = build_mil(opt)
    optimizer, infos = build_optimizer(opt, model, infos)
    crit = Criterion() # Training with RL, then add reward crit
    model.cuda()
    model.train()  # Assure in training mode
    return model, crit, optimizer, infos

def load_models(opt, infos):
    model = build_mil(opt)
    crit = Criterion(opt)
    return model, crit