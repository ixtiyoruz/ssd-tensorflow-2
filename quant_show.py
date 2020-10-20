#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Sep 11 08:54:47 2020

@author: essys
"""
import os
os.environ['CUDA_VISIBLE_DEVICES'] = '0'
import cv2
import tensorflow as tf
from tensorflow import keras
import numpy as np
import pathlib
import matplotlib.pyplot as plt
from box_utils import decode, compute_nms
import yaml
from anchor import generate_default_boxes, get_anchors, get_default_params
from scipy.special import softmax



tflite_model_file = "./ssd.quant.tflite"

interpreter = tf.lite.Interpreter(model_path=str(tflite_model_file))
interpreter.allocate_tensors()


input_index = interpreter.get_input_details()[0]["index"]
p_conf_index = interpreter.get_output_details()[0]["index"]
p_boxes_index = interpreter.get_output_details()[1]["index"]



img = cv2.imread("/home/essys/Pictures/picture.jpeg", -1)

img = cv2.resize(img, (300,300))
input_details = interpreter.get_input_details()[0]

input_scale, input_zero_point = input_details["quantization"]
test_image = img/127.0 -1.0
test_image = test_image / input_scale + input_zero_point
      
test_image = np.expand_dims(img, axis=0).astype(input_details["dtype"])      
interpreter.set_tensor(input_index, test_image)
interpreter.invoke()


p_conf = interpreter.get_tensor(p_conf_index)
p_boxes = interpreter.get_tensor(p_boxes_index)
# apply scaling dequantization
# real value = (int8val - zeropoint) x scale
p_conf = (p_conf - np.float32(interpreter.get_output_details()[0]['quantization'][1])) * np.float32(interpreter.get_output_details()[0]['quantization'][0])
p_boxes = (p_boxes - np.float32(interpreter.get_output_details()[1]['quantization'][1])) *np.float32( interpreter.get_output_details()[1]['quantization'][0])

with open('./config.yml') as f:
    cfg = yaml.load(f)

try:
    config = cfg['SSD300']#[args.arch.upper()]
except AttributeError:
    raise ValueError('Unknown architecture:')        
default_boxes = generate_default_boxes(config)
newres = decode(default_boxes, p_boxes[0]).numpy()
conf = softmax(p_conf, -1)[0]
classes = np.argmax(conf, -1)

# sort ant filter to threshold > 0.5, top 400 dets
def det_sort_filt(boxes, conf, classes, topn =100, threshold=0.5):
    # one class
    conf = conf[:, 1:]
    scores = np.squeeze(conf)
    
    #filter 1  classes
    mask1 = classes  == 1
    mask2 = scores >= threshold
    mask = np.logical_and(mask1, mask2)
    boxes = boxes[mask]
    scores = scores[mask]
    classes = classes[mask]
    # scores = np.max(conf, -1)
    # print(scores, conf)
    # sort
    idxes = np.argsort(-scores)
    idxes = idxes[:topn]
    # print(idxes)
    # conf =conf[idxes]
    classes =classes[idxes]
    scores =scores[idxes]
    boxes =boxes[idxes]
    
    
    return classes, scores, boxes
classes, scores, boxes = det_sort_filt(newres, conf, classes)
# now apply nms
def bboxes_nms(classes, scores, bboxes, nms_threshold=0.45):
    """Apply non-maximum selection to bounding boxes.
    """
    keep_bboxes = np.ones(scores.shape, dtype=np.bool)
    for i in range(scores.size-1):
        if keep_bboxes[i]:
            # Computer overlap with bboxes which are following.
            overlap = bboxes_jaccard(bboxes[i], bboxes[(i+1):])
            # Overlap threshold for keeping + checking part of the same class
            keep_overlap = np.logical_or(overlap < nms_threshold, classes[(i+1):] != classes[i])
            keep_bboxes[(i+1):] = np.logical_and(keep_bboxes[(i+1):], keep_overlap)

    idxes = np.where(keep_bboxes)
    return classes[idxes], scores[idxes], bboxes[idxes]


def bboxes_jaccard(bboxes1, bboxes2):
    """Computing jaccard index between bboxes1 and bboxes2.
    Note: bboxes1 and bboxes2 can be multi-dimensional, but should broacastable.
    """
    bboxes1 = np.transpose(bboxes1)
    bboxes2 = np.transpose(bboxes2)
    # Intersection bbox and volume.
    int_ymin = np.maximum(bboxes1[0], bboxes2[0])
    int_xmin = np.maximum(bboxes1[1], bboxes2[1])
    int_ymax = np.minimum(bboxes1[2], bboxes2[2])
    int_xmax = np.minimum(bboxes1[3], bboxes2[3])

    int_h = np.maximum(int_ymax - int_ymin, 0.)
    int_w = np.maximum(int_xmax - int_xmin, 0.)
    int_vol = int_h * int_w
    # Union volume.
    vol1 = (bboxes1[2] - bboxes1[0]) * (bboxes1[3] - bboxes1[1])
    vol2 = (bboxes2[2] - bboxes2[0]) * (bboxes2[3] - bboxes2[1])
    jaccard = int_vol / (vol1 + vol2 - int_vol)
    return jaccard

classes, scores, boxes = bboxes_nms(classes, scores, boxes, nms_threshold=0.45)

# classes, scores, boxes = ssd_bboxes_select_layer(classes, boxes, scores)


img = cv2.imread("/home/essys/Pictures/picture.jpeg", -1)

img = cv2.resize(img, (300,300))

def draw_image(img, bboxes, classes, mode = 'gt'):
    h,w,c = np.shape(img)
    for i in range(len(bboxes)):
            bbox = bboxes[i]
            x1,y1, x2,y2 = bbox 
            # x1,y1, x2,y2 = x1, y1-y2/2,x1 + x2/2,y1 +  y2/2
            x1,y1,x2,y2 = int(x1 * w), int(y1 * h), int(x2 * w), int(y2 * h)
            print(x1, x2, y1, y2, mode)
            cv2.rectangle(img, (x1, y1), (x2, y2), (255,0,0), 2)            
            cv2.rectangle(img, (x1, y1), (x1+10, y1+15), (0,0,0), -1) 
            if(type(classes[i]) in [np.int64, np.int32, np.int8, int]):                
                cv2.putText(img,str(classes[i]), (x1,y1+10),1, 1.0, (255,255,255))
            else:
                cv2.putText(img,str(classes[i].numpy()), (x1,y1+10),1, 1.0, (255,255,255))    

# NUM_CLASSES = 2
# newres = decode(default_boxes, newres).numpy()

# confs = tf.math.softmax(p_conf[0], axis=-1)
# classes = tf.math.argmax(confs, axis=-1)
# scores = tf.math.reduce_max(confs, axis=-1)
        
# out_boxes = []
# out_labels = []
# out_scores = []
# # print(confs.shape,classes.shape, scores.shape, boxes.shape)
# for c in range(1, NUM_CLASSES):
#     cls_scores = confs[:, c]

#     score_idx = cls_scores > 0.8
#     # cls_boxes = tf.boolean_mask(boxes, score_idx)
#     # cls_scores = tf.boolean_mask(cls_scores, score_idx)
#     cls_boxes = newres[score_idx]
#     cls_scores = cls_scores[score_idx]

#     nms_idx = compute_nms(cls_boxes, cls_scores, 0.35, 200)
#     cls_boxes = tf.gather(cls_boxes, nms_idx)
#     cls_scores = tf.gather(cls_scores, nms_idx)
#     cls_labels = [c] * cls_boxes.shape[0]

#     out_boxes.append(cls_boxes)
#     out_labels.extend(cls_labels)
#     out_scores.append(cls_scores)

# out_boxes = tf.concat(out_boxes, axis=0)
# out_scores = tf.concat(out_scores, axis=0)

# boxes = tf.clip_by_value(out_boxes, 0.0, 1.0).numpy()
# classes = np.array(out_labels)
# scores = out_scores.numpy()

draw_image(img, boxes, classes)
plt.imshow(img)
plt.show()