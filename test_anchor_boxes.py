#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Aug 26 09:28:03 2020

@author: essys
"""
from anchor import generate_default_boxes
import argparse
import tensorflow as tf
import os
import sys
import time
from box_utils import compute_target
import yaml
with open('./config.yml') as f:
    cfg = yaml.load(f)

try:
    config = cfg['SSD300']#[args.arch.upper()]
except AttributeError:
    pass
default_boxes = generate_default_boxes(config)

print(default_boxes)

import tensorflow_datasets as tfds

builder = tfds.builder('coco/' +'2017')
# 1. Create the tfrecord files (no-op if already exists)
builder.download_and_prepare()
# 2. Load the `tf.data.Dataset`
train_dataset = builder.as_dataset(split='train', shuffle_files=True)
val_dataset = builder.as_dataset(split='validation', shuffle_files=False)
info = builder.info
train_dataset = train_dataset

for i in range(10):
    sample = next(tfds.as_numpy(train_dataset.take(1)))
    print(sample['image/filename'])
    img = sample['image']
    boxes = sample['objects']['bbox']
    labels = sample['objects']['label']
    
    
    boxes = tf.constant(boxes, dtype=tf.float32)
    labels = tf.constant(labels, dtype=tf.int64)
    
    gt_confs, gt_locs = compute_target(
                    default_boxes, boxes, labels)