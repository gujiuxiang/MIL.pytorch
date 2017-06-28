"""Set up paths."""
import os
import os.path as osp
import sys
import platform
import cPickle
import cv2, numpy as np
from matplotlib.pyplot import show
import matplotlib.pyplot as plt
import urllib
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
from model import *

def add_path(path):
    if path not in sys.path:
        sys.path.insert(0, path)
        print 'added {}'.format(path)

this_dir = osp.dirname(__file__)
##############################################################################################
# METHOD #1: OpenCV, NumPy, and urllib
def url_to_image(url):
    # download the image, convert it to a NumPy array, and then read
    # it into OpenCV format
    resp = urllib.urlopen(url)
    image = np.asarray(bytearray(resp.read()), dtype="uint8")
    image = cv2.imdecode(image, cv2.IMREAD_COLOR)

    # return the image
    return image


def tic_toc_print(interval, string):
    global tic_toc_print_time_old
    if 'tic_toc_print_time_old' not in globals():
        tic_toc_print_time_old = time.time()
        print string
    else:
        new_time = time.time()
        if new_time - tic_toc_print_time_old > interval:
            tic_toc_print_time_old = new_time;
            print string


def save_variables(pickle_file_name, var, info, overwrite=False):
    """
      def save_variables(pickle_file_name, var, info, overwrite = False)
    """
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
    """
    d = load_variables(pickle_file_name)
    Output:
      d     is a dictionary of variables stored in the pickle file.
    """
    if os.path.exists(pickle_file_name):
        with open(pickle_file_name, 'rb') as f:
            d = cPickle.load(f)
        return d
    else:
        raise Exception('{:s} does not exists.'.format(pickle_file_name))


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

##############################################################################################

'''load vocabulary'''
vocab, functional_words, is_functional, pt = load_vocabulary()

parser = argparse.ArgumentParser(description='PyTorch MIL Training')
parser.add_argument('--start_from', type=str, default='')
parser.add_argument('--load_precision_score', type=str, default='model/precision_score.pkl')
parser.add_argument('--cnn_weight', default='model/mil.pth',
                    help='cnn weights')
opt = parser.parse_args()

mil_model = VGG_MIL(opt)
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
