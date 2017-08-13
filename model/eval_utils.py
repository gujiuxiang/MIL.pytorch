from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import torch
import torch.nn as nn
from torch.autograd import Variable

import numpy as np
import json
from json import encoder
import random
import string
import time
import os
import sys
import model.utils as utils

def eval_split(opt, model, crit, loader, eval_kwargs={}):
    verbose = eval_kwargs.get('verbose', True)
    num_images = eval_kwargs.get('num_images', eval_kwargs.get('val_images_use', -1))
    split = eval_kwargs.get('split', 'val')
    dataset = eval_kwargs.get('dataset', 'coco')

    # Make sure in the evaluation mode
    model.eval()
    loader.reset_iterator(split)

    n = 0
    loss = 0
    loss_sum = 0
    loss_evals = 1e-8
    predictions = []
    while True:
        data = loader.get_batch(split)
        n = n + loader.batch_size
        # forward the model to get loss
        tmp = [data['images'], data['mil_label']]
        tmp = [Variable(torch.from_numpy(_), volatile=True).cuda() for _ in tmp]
        images, mil_label = tmp

        loss = crit(model(images), mil_label).data[0]
        loss_sum = loss_sum + loss
        loss_evals = loss_evals + 1

        if data['bounds']['wrapped']:
            break
        if num_images >= 0 and n >= num_images:
            break

    # Switch back to training mode
    model.train()
    return loss_sum / loss_evals