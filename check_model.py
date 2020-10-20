# -*- coding: utf-8 -*-

import argparse
import tensorflow as tf
import os
import sys
import time
import yaml
import np_methods
from tensorflow.keras.optimizers.schedules import PiecewiseConstantDecay
from voc_data import VOCDataset
from datasets.COCOdataset import COCOdataset
from anchor import generate_default_boxes, get_anchors, get_default_params
from network_new import create_ssd
from losses import create_losses, ssd_losses
import tensorflow_model_optimization as tfmot
quantize_model = tfmot.quantization.keras.quantize_model
import numpy as np
from box_utils import decode, compute_nms
import cv2
import time 
#https://github.com/VisDrone/VisDrone-Dataset
parser = argparse.ArgumentParser()
parser.add_argument('--data-dir', default='/media/essys/bd21e577-e5db-47a3-8845-f27be3f083a4/dataset/voc/VOCdevkit/')
parser.add_argument('--data-year', default='2012')
parser.add_argument('--arch', default='SSD512')
parser.add_argument('--batch-size', default=1, type=int)
parser.add_argument('--num-batches', default=100, type=int)
parser.add_argument('--neg-ratio', default=3, type=int)
parser.add_argument('--initial-lr', default=7e-4, type=float)
parser.add_argument('--momentum', default=0.9, type=float)
parser.add_argument('--weight-decay', default=1e-5, type=float)
parser.add_argument('--num-epochs', default=120, type=int)
parser.add_argument('--checkpoint-dir', default='checkpoints')
parser.add_argument('--pretrained-type', default='base')
parser.add_argument('--gpu-id', default='2')
parser.add_argument('--freeze', default=1, type=int)
parser.add_argument('--quant', default=0, type=int)

args = parser.parse_args()

os.makedirs(args.checkpoint_dir, exist_ok=True)
os.environ['CUDA_VISIBLE_DEVICES'] = args.gpu_id

NUM_CLASSES = 2
with open('./config.yml') as f:
    cfg = yaml.load(f)

try:
    config = cfg['SSD512']
except AttributeError:
    raise ValueError('Unknown architecture: {}'.format(args.arch))
    
default_boxes = generate_default_boxes(config)
voc = COCOdataset('/media/essys/bd21e577-e5db-47a3-8845-f27be3f083a4/dataset/coco',default_boxes,batch_size=args.batch_size,datatype='train2017',
                  input_shape=[config['image_size'], config['image_size']], augmentation=['flip'])

voc_val = COCOdataset('/media/essys/bd21e577-e5db-47a3-8845-f27be3f083a4/dataset/coco',default_boxes,datatype='val2017',
                  input_shape=[config['image_size'], config['image_size']], augmentation=['none'])

try:
    ssd = create_ssd(NUM_CLASSES, args.arch,args.batch_size,
                    args.pretrained_type,
                    checkpoint_dir=args.checkpoint_dir,scope=None, freeze=args.freeze, quant=False)        
except Exception as e:
    print(e)
    print('The program is exiting...')
    sys.exit()



