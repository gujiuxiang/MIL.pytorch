#!/usr/bin/env python
#
# Copyright 2017 The Open Images Authors. All Rights Reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
# ==============================================================================
r"""Classifier inference utility.

This code takes a resnet_v1_101 checkpoint, runs the classifier on the image and
prints predictions in human-readable form.

-------------------------------
Example command:
-------------------------------

# 0. Create directory for model/data
WORK_PATH="/tmp/oidv2"
mkdir -p "${WORK_PATH}"
cd "${WORK_PATH}"

# 1. Download the model, inference code, and sample image
wget https://storage.googleapis.com/openimages/2017_07/classes-trainable.txt
wget https://storage.googleapis.com/openimages/2017_07/class-descriptions.csv
wget https://storage.googleapis.com/openimages/2017_07/oidv2-resnet_v1_101.ckpt.tar.gz
wget https://raw.githubusercontent.com/openimages/dataset/master/tools/classify_oidv2.py
tar -xzf oidv2-resnet_v1_101.ckpt.tar.gz

wget -O cat.jpg https://farm6.staticflickr.com/5470/9372235876_d7d69f1790_b.jpg

# 2. Run inference
python classify_oidv2.py \
--checkpoint_path='oidv2-resnet_v1_101.ckpt' \
--labelmap='classes-trainable.txt' \
--dict='class-descriptions.csv' \
--image="cat.jpg" \
--top_k=10 \
--score_threshold=0.3

# Sample output:
Image: "cat.jpg"

3272: /m/068hy - Pet (score = 0.96)
1076: /m/01yrx - Cat (score = 0.95)
0708: /m/01l7qd - Whiskers (score = 0.90)
4755: /m/0jbk - Animal (score = 0.90)
2847: /m/04rky - Mammal (score = 0.89)
2036: /m/0307l - Felidae (score = 0.79)
3574: /m/07k6w8 - Small to medium-sized cats (score = 0.77)
4799: /m/0k0pj - Nose (score = 0.70)
1495: /m/02cqfm - Close-up (score = 0.55)
0036: /m/012c9l - Domestic short-haired cat (score = 0.40)

-------------------------------
Note on image preprocessing:
-------------------------------

This is the code used to perform preprocessing:
--------
from preprocessing import preprocessing_factory

def PreprocessImage(image, network='resnet_v1_101', image_size=299):
  # If resolution is larger than 224 we need to adjust some internal resizing
  # parameters for vgg preprocessing.
  if any(network.startswith(x) for x in ['resnet', 'vgg']):
    preprocessing_kwargs = {
        'resize_side_min': int(256 * image_size / 224),
        'resize_side_max': int(512 * image_size / 224)
    }
  else:
    preprocessing_kwargs = {}
  preprocessing_fn = preprocessing_factory.get_preprocessing(
      name=network, is_training=False)

  height = image_size
  width = image_size
  image = preprocessing_fn(image, height, width, **preprocessing_kwargs)
  image.set_shape([height, width, 3])
  return image
--------

Note that there appears to be a small difference between the public version
of slim image processing library and the internal version (which the meta
graph is based on). Results that are very close, but not exactly identical to
that of the metagraph.
"""
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
import tensorflow as tf
import argparse
import random
import numpy as np
import time, os, sys
import json
import cv2
import os
import six.moves.urllib as urllib
import sys
import tarfile
import zipfile

from collections import defaultdict
from io import StringIO
from matplotlib import pyplot as plt
from PIL import Image

# This is needed since the notebook is stored in the object_detection folder.
flags = tf.app.flags
FLAGS = flags.FLAGS

def load_image_ids(split_name):
    ''' Load a list of (path,image_id tuples). Modify this to suit your data locations. '''
    split = []
    if split_name == 'coco_test2014':
      with open('data/mscoco/annotations/image_info_test2014.json') as f:
        data = json.load(f)
        for item in data['images']:
          image_id = int(item['id'])
          filepath = os.path.join('data/mscoco/test2014/', item['file_name'])
          split.append((filepath,image_id))
    elif split_name == 'coco_val2014':
      with open('data/mscoco/annotations/captions_val2014.json') as f:
        data = json.load(f)
        for item in data['images']:
          image_id = int(item['id'])
          filepath = os.path.join('data/mscoco/val2014/', item['file_name'])
          split.append((filepath,image_id))
    elif split_name == 'coco_train2014':
      with open('data/mscoco/annotations/captions_train2014.json') as f:
        data = json.load(f)
        for item in data['images']:
          image_id = int(item['id'])
          filepath = os.path.join('data/mscoco/train2014/', item['file_name'])
          split.append((filepath,image_id))
    elif split_name == 'coco_test2015':
      with open('data/mscoco/annotations/image_info_test2015.json') as f:
        data = json.load(f)
        for item in data['images']:
          image_id = int(item['id'])
          filepath = os.path.join('data/mscoco/test2015/', item['file_name'])
          split.append((filepath,image_id))
    elif split_name == 'genome':
      with open('data/visualgenome/image_data.json') as f:
        for item in json.load(f):
          image_id = int(item['image_id'])
          filepath = os.path.join('data/visualgenome/', item['url'].split('rak248/')[-1])
          split.append((filepath,image_id))
    elif split_name == 'chinese':
      with open('data/aic_i2t/ai_challenger_caption_train_20170902/caption_train_annotations_20170902.json') as f:
        for item in json.load(f):
          image_id = item['image_id']
          filepath = os.path.join('data/aic_i2t/ai_challenger_caption_train_20170902/caption_train_images_20170902', image_id)
          split.append((filepath,image_id))
    elif split_name == 'chinese_val':
      with open('data/aic_i2t/ai_challenger_caption_validation_20170910/caption_validation_annotations_20170910.json') as f:
        for item in json.load(f):
          image_id = item['image_id']
          filepath = os.path.join('data/aic_i2t/ai_challenger_caption_validation_20170910/caption_validation_images_20170910', image_id)
          split.append((filepath,image_id))
    elif split_name == 'chinese_test1':
      with open('data/aic_i2t/ai_challenger_caption_test1_20170923/caption_test1_annotations_20170923.json') as f:
        for item in json.load(f):
          image_id = item['image_id']
          filepath = os.path.join('data/aic_i2t/ai_challenger_caption_test1_20170923/caption_test1_images_20170923', image_id)
          split.append((filepath,image_id))
    else:
      print('Unknown split')
    return split

def get_classifications_from_im(args, g, sess, image_ids):
    save_dir = 'oid_data/'
    input_values = g.get_tensor_by_name('input_values:0')
    predictions = g.get_tensor_by_name('multi_predictions:0')
    count = 0
    for im_file, image_id in image_ids:
        compressed_image = tf.gfile.FastGFile(im_file, 'rb').read()
        predictions_eval = sess.run(predictions, feed_dict={input_values: [compressed_image]})
        if 'chinese' in args.data_split:
            np.savez_compressed(save_dir + 'aic_i2t/oid_cls/' + str(image_id), feat=predictions_eval)
        else:
            np.savez_compressed(save_dir + 'mscoco/oid_cls/' + str(image_id), feat=predictions_eval)
        if (count % 100) == 0:
            print('{:d}'.format(count + 1))
        count += 1

def parse_args():
    """
    Parse input arguments
    """
    parser = argparse.ArgumentParser(description='Generate bbox output from a Fast R-CNN network')
    parser.add_argument('--gpu', dest='gpu_id', help='GPU id(s) to use', default='0', type=str)
    parser.add_argument('--type', dest='type', help='', default='det', type=str)
    parser.add_argument('--def', dest='prototxt', help='prototxt file defining the network', default='../models/vg/ResNet-101/faster_rcnn_end2end_final/test.prototxt', type=str)
    parser.add_argument('--net', dest='caffemodel', help='model to use', default='../data/faster_rcnn_models/resnet101_faster_rcnn_final.caffemodel', type=str)
    parser.add_argument('--out', dest='outfile', help='output filepath', default='karpathy_train_resnet101_faster_rcnn_genome', type=str)
    parser.add_argument('--cfg', dest='cfg_file', help='optional config file', default='../experiments/cfgs/faster_rcnn_end2end_resnet.yml', type=str)
    parser.add_argument('--split', dest='data_split', help='dataset to use', default='coco_val2014', type=str)
    parser.add_argument('--checkpoint_path', dest='checkpoint_path', help='', default='model/oidv2_resnet_v1_101/oidv2-resnet_v1_101.ckpt', type=str)
    parser.add_argument('--set', dest='set_cfgs', help='set config keys', default=None, nargs=argparse.REMAINDER)

    args = parser.parse_args()
    return args


def main(_):
    args = parse_args()

    print('Called with args:')
    print(args)
    g = tf.Graph()
    with g.as_default():
        with tf.Session() as sess:
            saver = tf.train.import_meta_graph(args.checkpoint_path + '.meta')
            saver.restore(sess, args.checkpoint_path)
            image_ids = load_image_ids(args.data_split)
            get_classifications_from_im(args, g, sess, image_ids)

if __name__ == '__main__':
  tf.app.run()
