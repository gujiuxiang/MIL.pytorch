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
import itertools

'''
-- Learning rate annealing schedule. We will build a new optimizer for
-- each epoch.
--
-- By default we follow a known recipe for a 55-epoch training. If
-- the learningRate command-line parameter has been specified, though,
-- we trust the user is doing something manual, and will use her
-- exact settings for all optimization.
--
-- Return values:
--    diff to apply to optimState,
--    true IFF this is the first epoch of a new regime
'''
def set_lr(optimizer, lr):
    for group in optimizer.param_groups:
        group['lr'] = lr

def set_weightDecay(optimizer, weightDecay):
    for group in optimizer.param_groups:
        group['weight_decay'] = weightDecay

def update_lr(opt, epoch, optimizer):
    if epoch > opt.learning_rate_decay_start and opt.learning_rate_decay_start >= 0:
        frac = (epoch - opt.learning_rate_decay_start) // opt.learning_rate_decay_every
        decay_factor = opt.learning_rate_decay_rate ** frac
        opt.current_lr = opt.learning_rate * decay_factor
        set_lr(optimizer, opt.current_lr)  # set the decayed rate
    else:
        opt.current_lr = opt.learning_rate

def paramsForEpoch(opt, epoch, optimizer):
    # start, end,    LR,   WD,
    regimes = [ [ 0,     5,    1e-2,   5e-4, ],
                [ 6,     10,   5e-3,   5e-4  ],
                [ 11,    20,   1e-3,   0 ],
                [ 31,    30,   5e-4,   0 ],
                [ 31,    1e8,  1e-4,   0 ]]
    for row in regimes:
        if epoch>=row[0] and epoch<= row[1]:
            learningRate = row[2]
            weightDecay = row[3]
            opt.learning_rate = learningRate
            opt.weight_decay = weightDecay
            set_lr(optimizer, learningRate)
            set_weightDecay(optimizer, weightDecay)

def add_summary_value(writer, key, value, iteration):
    summary = tf.Summary(value=[tf.Summary.Value(tag=key, simple_value=value)])
    writer.add_summary(summary, iteration)

def history_infos(opt):
    infos = {}
    if len(opt.start_from) != 0:  # open old infos and check if models are compatible
        model_id = opt.start_from
        infos_id = model_id.replace('save/', '') + '.infos-best.pkl'
        with open(os.path.join(opt.start_from, infos_id)) as f:
            infos = cPickle.load(f)
            saved_model_opt = infos['opt']

    iteration = infos.get('iter', 0)
    epoch = infos.get('epoch', 0)
    val_result_history = infos.get('val_result_history', {})
    loss_history = infos.get('loss_history', {})
    lr_history = infos.get('lr_history', {})
    best_val_score = infos.get('best_val_score', None) if opt.load_best_score == 1 else 0
    val_loss = 0.0
    val_history = [val_result_history, best_val_score, val_loss]
    train_history = [loss_history, lr_history]
    return opt, infos, iteration, epoch, val_history, train_history


def add_path(path):
    if path not in sys.path:
        sys.path.insert(0, path)
        print('added {}'.format(path))

def save_variables(pickle_file_name, var, info, overwrite=False):
    if os.path.exists(pickle_file_name) and overwrite == False:
        raise Exception('{:s} exists and over write is false.'.format(pickle_file_name))
    # Construct the dictionary
    assert (type(var) == list);
    assert (type(info) == list);
    d = {}
    for i in xrange(len(var)):
        d[info[i]] = var[i]
    with open(pickle_file_name, 'wb') as f:
        cPickle.dump(d, f, cPickle.HIGHEST_PROTOCOL)


def load_variables(pickle_file_name):
    # d     is a dictionary of variables stored in the pickle file.
    if os.path.exists(pickle_file_name):
        with open(pickle_file_name, 'rb') as f:
            d = cPickle.load(f)
        return d
    else:
        raise Exception('{:s} does not exists.'.format(pickle_file_name))

def to_contiguous(tensor):
    if tensor.is_contiguous():
        return tensor
    else:
        return tensor.contiguous()

def clip_gradient(optimizer, grad_clip):
    for group in optimizer.param_groups:
        for param in group['params']:
            param.grad.data.clamp_(-grad_clip, grad_clip)

# METHOD #1: OpenCV, NumPy, and urllib
def url_to_image(url):
    # download the image, convert it to a NumPy array, and then read
    # it into OpenCV format
    resp = urllib.urlopen(url)
    image = np.asarray(bytearray(resp.read()), dtype="uint8")
    image = cv2.imdecode(image, cv2.IMREAD_COLOR)

    # return the image
    return image

def upsample_image(im, sz):
    h = im.shape[0]
    w = im.shape[1]
    s = np.float(max(h, w))
    I_out = np.zeros((sz, sz, 3), dtype=np.float);
    I = cv2.resize(im, None, None, fx=np.float(sz) / s, fy=np.float(sz) / s, interpolation=cv2.INTER_LINEAR);
    SZ = I.shape;
    I_out[0:I.shape[0], 0:I.shape[1], :] = I;
    return I_out, I, SZ

def compute_precision_score_mapping_torch(thresh, prec, score):
    thresh, ind_thresh = torch.sort(torch.from_numpy(thresh), 0, descending=False)

    prec, ind_prec = torch.sort(torch.from_numpy(prec), 0, descending=False)
    val = None
    return val

def compute_precision_mapping(pt):
    thresh_all = []
    prec_all = []
    for jj in xrange(1000):
        thresh = pt['details']['score'][:, jj]
        prec = pt['details']['precision'][:, jj]
        ind = np.argsort(thresh); # thresh, ind = torch.sort(thresh)
        thresh = thresh[ind];
        indexes = np.unique(thresh, return_index=True)[1]
        indexes = np.sort(indexes);
        thresh = thresh[indexes]

        thresh = np.vstack((min(-1000, min(thresh) - 1), thresh[:, np.newaxis], max(1000, max(thresh) + 1)));

        prec = prec[ind];
        for i in xrange(1, len(prec)):
            prec[i] = max(prec[i], prec[i - 1]);
        prec = prec[indexes]

        prec = np.vstack((prec[0], prec[:, np.newaxis], prec[-1]));
        thresh_all.append(thresh)
        prec_all.append(prec)
    precision_score = {'thresh': thresh_all, "prec": prec_all}
    return precision_score

def compute_precision_score_mapping(thresh, prec, score):
    ind = np.argsort(thresh); # thresh, ind = torch.sort(thresh)
    thresh = thresh[ind];
    indexes = np.unique(thresh, return_index=True)[1]
    indexes = np.sort(indexes);
    thresh = thresh[indexes]

    thresh = np.vstack((min(-1000, min(thresh) - 1), thresh[:, np.newaxis], max(1000, max(thresh) + 1)));

    prec = prec[ind];
    for i in xrange(1, len(prec)):
        prec[i] = max(prec[i], prec[i - 1]);
    prec = prec[indexes]

    prec = np.vstack((prec[0], prec[:, np.newaxis], prec[-1]));

    f = interp1d(thresh[:, 0], prec[:, 0])
    val = f(score)
    return val

def load_vocabulary():
    # Load the vocabulary
    vocab_file = os.getcwd()+'/vocabs/vocab_train.pkl'
    vocab = load_variables(vocab_file)

    # define functional words
    functional_words = ['a', 'on', 'of', 'the', 'in', 'with', 'and', 'is', 'to', 'an', 'two', 'at', 'next', 'are']
    is_functional = np.array([x not in functional_words for x in vocab['words']])

    # load the score precision mapping file
    eval_file = os.getcwd()+'/model/coco_valid1_eval.pkl'
    pt = load_variables(eval_file)
    return vocab, functional_words, is_functional, pt

def tic_toc_print(interval, string):
    global tic_toc_print_time_old
    if 'tic_toc_print_time_old' not in globals():
        tic_toc_print_time_old = time.time()
        print(string)
    else:
        new_time = time.time()
        if new_time - tic_toc_print_time_old > interval:
            tic_toc_print_time_old = new_time;
            print(string)

