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
sys.path.append("/home/jxgu/github/MIL.pytorch/misc/models/research")
from utils import label_map_util
from utils import visualization_utils as vis_util
import utils.ops as utils_ops
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

def run_inference_for_single_image(image, graph):
  with graph.as_default():
    with tf.Session() as sess:
      # Get handles to input and output tensors
      ops = tf.get_default_graph().get_operations()
      all_tensor_names = {output.name for op in ops for output in op.outputs}
      tensor_dict = {}
      for key in [
          'num_detections', 'detection_boxes', 'detection_scores',
          'detection_classes', 'detection_masks'
      ]:
        tensor_name = key + ':0'
        if tensor_name in all_tensor_names:
          tensor_dict[key] = tf.get_default_graph().get_tensor_by_name(
              tensor_name)
      if 'detection_masks' in tensor_dict:
        # The following processing is only for single image
        detection_boxes = tf.squeeze(tensor_dict['detection_boxes'], [0])
        detection_masks = tf.squeeze(tensor_dict['detection_masks'], [0])
        # Reframe is required to translate mask from box coordinates to image coordinates and fit the image size.
        real_num_detection = tf.cast(tensor_dict['num_detections'][0], tf.int32)
        detection_boxes = tf.slice(detection_boxes, [0, 0], [real_num_detection, -1])
        detection_masks = tf.slice(detection_masks, [0, 0, 0], [real_num_detection, -1, -1])
        detection_masks_reframed = utils_ops.reframe_box_masks_to_image_masks(
            detection_masks, detection_boxes, image.shape[0], image.shape[1])
        detection_masks_reframed = tf.cast(tf.greater(detection_masks_reframed, 0.2), tf.uint8)
        # Follow the convention by adding back the batch dimension
        tensor_dict['detection_masks'] = tf.expand_dims(
            detection_masks_reframed, 0)
      image_tensor = tf.get_default_graph().get_tensor_by_name('image_tensor:0')

      # Run inference
      output_dict = sess.run(tensor_dict, feed_dict={image_tensor: np.expand_dims(image, 0)})

      # all outputs are float32 numpy arrays, so convert types as appropriate
      output_dict['num_detections'] = int(output_dict['num_detections'][0])
      output_dict['detection_classes'] = output_dict[
          'detection_classes'][0].astype(np.uint8)
      output_dict['detection_boxes'] = output_dict['detection_boxes'][0]
      output_dict['detection_scores'] = output_dict['detection_scores'][0]
      if 'detection_masks' in output_dict:
        output_dict['detection_masks'] = output_dict['detection_masks'][0]
  return output_dict

def get_detections_from_im(args, detection_graph, image_ids):
    save_dir = '/home/jxgu/github/MIL.pytorch/oid_data/'
    count = 0
    for im_file, image_id in image_ids:
        image = Image.open(im_file)
        # the array based representation of the image will be used later in order to prepare the
        # result image with boxes and labels on it.
        image_np = load_image_into_numpy_array(image)
        # Expand dimensions since the model expects images to have shape: [1, None, None, 3]
        image_np_expanded = np.expand_dims(image_np, axis=0)
        # Actual detection.
        output_dict = run_inference_for_single_image(image_np, detection_graph)
        if 'chinese' in args.data_split:
            np.savez_compressed(save_dir + 'aic_i2t/oid_det/' + str(image_id), feat=output_dict)
        else:
            np.savez_compressed(save_dir + 'mscoco/oid_det/' + str(image_id), feat=output_dict)

        if (count % 100) == 0:
            print('{:d}'.format(count + 1))
        count += 1

def load_image_into_numpy_array(image):
  (im_width, im_height) = image.size
  return np.array(image.getdata()).reshape(
      (im_height, im_width, 3)).astype(np.uint8)

def parse_args():
    """
    Parse input arguments
    """
    parser = argparse.ArgumentParser(description='Generate bbox output from a Fast R-CNN network')
    parser.add_argument('--gpu', dest='gpu_id', help='GPU id(s) to use', default='0', type=str)
    parser.add_argument('--type', dest='type', help='', default='det', type=str)
    parser.add_argument('--def', dest='prototxt', help='prototxt file defining the network', default='../models/vg/ResNet-101/faster_rcnn_end2end_final/test.prototxt', type=str)
    parser.add_argument('--net', dest='caffemodel', help='model to use', default='../../../../data/faster_rcnn_models/resnet101_faster_rcnn_final.caffemodel', type=str)
    parser.add_argument('--out', dest='outfile', help='output filepath', default='karpathy_train_resnet101_faster_rcnn_genome', type=str)
    parser.add_argument('--cfg', dest='cfg_file', help='optional config file', default='../experiments/cfgs/faster_rcnn_end2end_resnet.yml', type=str)
    parser.add_argument('--split', dest='data_split', help='dataset to use', default='coco_train2014', type=str)
    parser.add_argument('--checkpoint_path', dest='checkpoint_path', help='', default='/home/jxgu/github/MIL.pytorch/model/faster_rcnn_inception_resnet_v2_atrous_oid_2018_01_28.tar.gz', type=str)
    parser.add_argument('--set', dest='set_cfgs', help='set config keys', default=None, nargs=argparse.REMAINDER)

    args = parser.parse_args()
    return args


def main(_):
    args = parse_args()

    # What model to download.
    MODEL_NAME = '/home/jxgu/github/MIL.pytorch/model/faster_rcnn_inception_resnet_v2_atrous_oid_2018_01_28'
    MODEL_FILE = MODEL_NAME + '.tar.gz'
    PATH_TO_CKPT = MODEL_NAME + '/frozen_inference_graph.pb'
    tar_file = tarfile.open(MODEL_FILE)
    for file in tar_file.getmembers():
        file_name = os.path.basename(file.name)
        if 'frozen_inference_graph.pb' in file_name:
            tar_file.extract(file, os.getcwd())

    tar_file = tarfile.open(args.checkpoint_path)
    for file in tar_file.getmembers():
        file_name = os.path.basename(file.name)
        if 'frozen_inference_graph.pb' in file_name:
            tar_file.extract(file, os.getcwd())

    detection_graph = tf.Graph()
    with detection_graph.as_default():
        od_graph_def = tf.GraphDef()
        with tf.gfile.GFile(PATH_TO_CKPT, 'rb') as fid:
            serialized_graph = fid.read()
            od_graph_def.ParseFromString(serialized_graph)
            tf.import_graph_def(od_graph_def, name='')
            image_ids = load_image_ids(args.data_split)
            get_detections_from_im(args, detection_graph, image_ids)
if __name__ == '__main__':
  tf.app.run()
