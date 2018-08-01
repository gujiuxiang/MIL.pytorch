"""Set up paths."""
import os
import os.path as osp
import sys
import platform
import cPickle
import cv2, numpy as np
from matplotlib.pyplot import show
import matplotlib.pyplot as plt
from scipy.interpolate import interp1d
import time
import Image
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
import torch.backends.cudnn as cudnn
import tensorflow as tf
import torchvision
import torchvision.transforms as transforms
from six.moves import cPickle
import gc
import os
import pickle
import argparse
from model.models import *
from model.utils import *
import coco_voc

this_dir = osp.dirname(__file__)
##############################################################################################

def test_img(im, net, base_image_size, means):
    """
    Calls Caffe to get output for this image
    """
    batch_size = 1
    # Resize image
    im_orig = im.astype(np.float32, copy=True)
    im_orig -= means

    im, gr, grr = upsample_image(im_orig, base_image_size)
    im = np.transpose(im, axes=(2, 0, 1))
    im = im[np.newaxis, :, :, :]

    # Pass into model
    mil_prob = net(Variable(torch.from_numpy(im), requires_grad=False).cuda())
    return mil_prob


def output_words_image(threshold_metric, output_metric, min_words, threshold, vocab, is_functional):
    ind_output = np.argsort(threshold_metric)
    ind_output = ind_output[::-1]
    must_keep1 = threshold_metric[ind_output] >= threshold;
    must_keep2 = np.cumsum(is_functional[ind_output]) < 1 + min_words;
    ind_output = [ind for j, ind in enumerate(ind_output) if must_keep1[j] or must_keep2[j]]
    out = [(vocab['words'][ind], output_metric[ind], threshold_metric[ind]) for ind in ind_output]
    return out

##############################################################################################

'''load vocabulary'''
vocab, functional_words, is_functional, pt = load_vocabulary()

parser = argparse.ArgumentParser(description='PyTorch MIL Training')
parser.add_argument('--start_from', type=str, default='')
parser.add_argument('--load_precision_score', type=str, default='')
parser.add_argument('--cnn_weight', default='model/mil.pth',
                    help='cnn weights')
opt = parser.parse_args()

mil_model = vgg_mil(opt)
mil_model.cuda()
mil_model.eval()


'''load caffe model'''
mean = np.array([[[103.939, 116.779, 123.68]]]);
base_image_size = 565;

'''Load the image'''
imageurl = 'http://img1.10bestmedia.com/Images/Photos/333810/Montrose_54_990x660.jpg'
im = url_to_image(imageurl)
im = cv2.resize(im, (base_image_size, base_image_size), interpolation=cv2.INTER_CUBIC)

# Run the model
mil_prob = test_img(im, mil_model, base_image_size, mean)
mil_prob = mil_prob.data.cpu().float().numpy()
# Compute precision mapping - slow in per image mode, much faster in batch mode
prec = np.zeros(mil_prob.shape)
if len(opt.load_precision_score) >0 :
    precision_score = pickle.load(open(opt.load_precision_score, 'rb'))
else:
    precision_score = compute_precision_mapping(pt)

for jj in xrange(prec.shape[1]):
    f = interp1d(precision_score['thresh'][jj][:,0], precision_score['prec'][jj][:,0])
    #prec[:, jj] = f(mil_prob[:, jj])
    prec[:, jj] = mil_prob[:, jj]
mil_prec = prec

#cv2.imshow('image', im)
# Output words
out = output_words_image(mil_prec[0, :], mil_prec[0, :], \
                         min_words=10, threshold=0.0, vocab=vocab, is_functional=is_functional)

plt.rcParams['figure.figsize'] = (10, 10)
plt.imshow(im[:, :, [2, 1, 0]])
plt.gca().set_axis_off()
det_atts = []
det_atts_w = []
index = 0
for (a, b, c) in out:
    if a not in functional_words:
        if index < 10:
            det_atts.append(a)
            det_atts_w.append(np.round(b, 2))
            index = index + 1
            # print '{:s} [{:.2f}, {:.2f}]   '.format(a, np.round(b,2), np.round(c,2))

print det_atts
print det_atts_w
