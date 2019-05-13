from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
from six.moves import cPickle
import json
import h5py
import os
import numpy as np
import random
import torch
import cv2, numpy as np
from torchvision import transforms as trn
from multiprocessing.dummy import Pool
import math
import gc

preprocess = trn.Compose([
        #trn.ToTensor(),
        trn.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
])

preprocess_vgg16 = trn.Compose([
        #trn.ToTensor(),
        trn.Normalize([123.680, 103.939, 116.779], [1.000, 1.000, 1.000])
])

def upsample_image(im, sz):
    h = im.shape[0]
    w = im.shape[1]
    s = np.float(max(h, w))
    #I_out = np.zeros((sz, sz, 3), dtype=np.float);
    #I = cv2.resize(im, None, None, fx=np.float(sz) / s, fy=np.float(sz) / s, interpolation=cv2.INTER_CUBIC); #INTER_CUBIC, INTER_LINEAR
    I = cv2.resize(im, (sz, sz), interpolation=cv2.INTER_LINEAR)
    SZ = I.shape;
    #I_out[0:I.shape[0], 0:I.shape[1], :] = I;
    return I, I, SZ

def preprocess_vgg19_mil(Image):
    if len(Image.shape) == 2:
        Image = Image[:, :, np.newaxis]
        Image = np.concatenate((Image, Image, Image), axis=2)

    mean = np.array([[[103.939, 116.779, 123.68]]]);
    base_image_size = 565;
    Image = cv2.resize(np.transpose(Image, axes=(1, 2, 0)), (base_image_size, base_image_size), interpolation=cv2.INTER_CUBIC)
    Image_orig = Image.astype(np.float32, copy=True)
    Image_orig -= mean
    im = Image_orig
    #im, gr, grr = upsample_image(Image_orig, base_image_size)
    # im = cv2.resize(Image_orig, (base_image_size, base_image_size), interpolation=cv2.INTER_CUBIC)
    im = np.transpose(im, axes=(2, 0, 1))
    im = im[np.newaxis, :, :, :]
    return im

'''
Load data from h5 files
'''
class DataLoader():
    def reset_iterator(self, split):
        # if load files from directory, then reset the prefetch process
        self.iterators[split] = 0

    def get_vocab_size(self):
        return self.vocab_size

    def get_vocab(self):
        return self.ix_to_word

    def get_seq_length(self):
        return self.seq_length

    def __init__(self, opt):
        self.type = 'h5'
        self.opt = opt
        self.model = getattr(opt, 'model', 'resnet101')
        self.attrs_in = getattr(opt, 'attrs_in', 0)
        self.attrs_out = getattr(opt, 'attrs_out', 0)
        self.att_im = getattr(opt, 'att_im', 1)
        self.pre_ft = getattr(opt, 'pre_ft', 1)
        self.mil_vocab_outsize = 1000
        self.top_attrs = 10
        self.fc_feat_size = opt.fc_feat_size
        self.att_feat_size = opt.att_feat_size
        self.batch_size = opt.batch_size
        self.seq_per_img = opt.seq_per_img

        # load the json file which contains additional information about the dataset
        print('DataLoader loading json file: ', opt.input_json)
        self.info = json.load(open(self.opt.input_json))
        self.mil_vocab = cPickle.load(open('model/mil_vocab.pkl'))
        self.ix_to_word = self.info['ix_to_word']
        self.vocab_size = len(self.ix_to_word)
        print('vocab size is ', self.vocab_size)

        # open the hdf5 file
        print('DataLoader loading h5 file: ', opt.input_im_h5)
        self.h5_im_file = h5py.File(self.opt.input_im_h5)
        # extract image size from dataset
        images_size = self.h5_im_file['images'].shape
        assert len(images_size) == 4, 'images should be a 4D tensor'
        assert images_size[2] == images_size[3], 'width and height must match'
        self.num_images = images_size[0]
        self.num_channels = images_size[1]
        self.max_image_size = images_size[2]
        print('read %d images of size %dx%dx%d' %(self.num_images,
                    self.num_channels, self.max_image_size, self.max_image_size))
        # load in the sequence data
        self.h5_label_file = h5py.File(self.opt.input_label_h5, 'r', driver='core')
        seq_size = self.h5_label_file['labels'].shape
        self.seq_length = seq_size[1]
        semantic_attrs_size = self.h5_label_file['semantic_words'].shape
        self.semantic_attrs_length = semantic_attrs_size[1]
        print('max sequence length in data is', self.seq_length)
        print('max semantic words length in data is', self.semantic_attrs_length)
        # load the pointers in full to RAM (should be small enough)
        self.label_start_ix = self.h5_label_file['label_start_ix'][:]
        self.label_end_ix = self.h5_label_file['label_end_ix'][:]

        self.num_images = self.label_start_ix.shape[0]
        print('read %d image / features' %(self.num_images))

        # separate out indexes for each of the provided splits
        self.split_ix = {'train': [], 'val': [], 'test': []}
        for ix in range(len(self.info['images'])):
            img = self.info['images'][ix]
            if img['split'] == 'train':
                self.split_ix['train'].append(ix)
            elif img['split'] == 'val':
                self.split_ix['val'].append(ix)
            elif img['split'] == 'test':
                self.split_ix['test'].append(ix)
            elif opt.train_only == 0: # restval
                self.split_ix['train'].append(ix)

        print('assigned %d images to split train' %len(self.split_ix['train']))
        print('assigned %d images to split val' %len(self.split_ix['val']))
        print('assigned %d images to split test' %len(self.split_ix['test']))

        self.iterators = {'train': 0, 'val': 0, 'test': 0}

    def gen_mil_gt(self, attrs):
        mil_batch = np.zeros([1, self.mil_vocab_outsize], dtype='int')
        for k in range(len(attrs)):
            if attrs[k] > 0:
                for i in range(self.mil_vocab_outsize):
                    if self.ix_to_word[str(attrs[k])] == self.mil_vocab[i]:
                        mil_batch[0, i] = 1

        return mil_batch

    def get_batch(self, split, batch_size=None, seq_per_img=None):
        split_ix = self.split_ix[split]
        batch_size = batch_size or self.batch_size
        seq_per_img = seq_per_img or self.seq_per_img

        if 'vgg19' in self.model:
            img_batch = np.ndarray([batch_size, 3, 565, 565], dtype='float32')
        else:
            img_batch = np.ndarray([batch_size, 3, 224, 224], dtype='float32')
        label_batch = np.zeros([batch_size * self.seq_per_img, self.seq_length + 2], dtype = 'int')
        mask_batch = np.zeros([batch_size * self.seq_per_img, self.seq_length + 2], dtype = 'float32')
        attrs_batch = np.zeros([batch_size, self.top_attrs], dtype = 'int')
        mil_batch = np.zeros([batch_size, self.mil_vocab_outsize], dtype='int')
        max_index = len(split_ix)
        wrapped = False

        infos = []
        gts = []

        for i in range(batch_size):
            import time
            t_start = time.time()

            ri = self.iterators[split]
            ri_next = ri + 1
            if ri_next >= max_index:
                ri_next = 0
                wrapped = True
            self.iterators[split] = ri_next
            ix = split_ix[ri]

            #img = self.load_image(self.image_info[ix]['filename'])
            img = self.h5_im_file['images'][ix, :, :, :]
            if 'resnet' in self.model:
                img_batch[i] = preprocess(torch.from_numpy(img[:, 16:-16, 16:-16].astype('float32')/255.0)).numpy()
            else:
                #img_batch[i] = preprocess_vgg16(torch.from_numpy(img[:, 16:-16, 16:-16].astype('float32'))).numpy()
                img_batch[i] = preprocess_vgg19_mil(img)

            # fetch the semantic_attributes
            attrs_batch[i] = self.h5_label_file['semantic_words'][ix, : self.top_attrs]
            mil_batch[i] = self.gen_mil_gt(attrs_batch[i])

            # fetch the sequence labels
            ix1 = self.label_start_ix[ix] - 1 #label_start_ix starts from 1
            ix2 = self.label_end_ix[ix] - 1
            ncap = ix2 - ix1 + 1 # number of captions available for this image
            assert ncap > 0, 'an image does not have any label. this can be handled but right now isn\'t'

            # record associated info as well
            info_dict = {}
            info_dict['ix'] = ix
            info_dict['id'] = self.info['images'][ix]['id']
            info_dict['file_path'] = self.info['images'][ix]['file_path']
            infos.append(info_dict)

        # generate mask
        t_start = time.time()
        nonzeros = np.array(map(lambda x: (x != 0).sum()+2, label_batch))
        for ix, row in enumerate(mask_batch):
            row[:nonzeros[ix]] = 1

        data = {}

        data['images'] = img_batch # if pre_ft is 1, then it equals None
        data['semantic_words'] = attrs_batch # if attributes is 1, then it equals None
        data['mil_label'] = mil_batch
        data['bounds'] = {'it_pos_now': self.iterators[split], 'it_max': len(split_ix), 'wrapped': wrapped}
        data['infos'] = infos

        gc.collect()

        return data