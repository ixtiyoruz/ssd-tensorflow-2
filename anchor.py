import itertools
import math
import tensorflow as tf
import numpy as np
from collections import namedtuple
def generate_default_boxes(config):
    """ Generate default boxes for all feature maps

    Args:
        config: information of feature maps
            scales: boxes' size relative to image's size
            fm_sizes: sizes of feature maps
            ratios: box ratios used in each feature maps

    Returns:
        default_boxes: tensor of shape (num_default, 4)
                       with format (cx, cy, w, h)
    """
    default_boxes = []
    scales = config['scales']
    fm_sizes = config['fm_sizes']
    ratios = config['ratios']

    for m, fm_size in enumerate(fm_sizes):
        for i, j in itertools.product(range(fm_size), repeat=2):
            cx = (j + 0.5) / fm_size
            cy = (i + 0.5) / fm_size
            default_boxes.append([
                cx,
                cy,
                scales[m],
                scales[m]
            ])

            default_boxes.append([
                cx,
                cy,
                math.sqrt(scales[m] * scales[m + 1]),
                math.sqrt(scales[m] * scales[m + 1])
            ])

            for ratio in ratios[m]:
                r = math.sqrt(ratio)
                default_boxes.append([
                    cx,
                    cy,
                    scales[m] * r,
                    scales[m] / r
                ])

                default_boxes.append([
                    cx,
                    cy,
                    scales[m] / r,
                    scales[m] * r
                ])

    default_boxes = tf.constant(default_boxes)
    default_boxes = tf.clip_by_value(default_boxes, 0.0, 1.0)

    return default_boxes



def ssd_anchor_one_layer(img_shape,
                         feat_shape,
                         sizes,
                         ratios,
                         step,
                         offset=0.5,
                         dtype=np.float32):
    """Computer SSD default anchor boxes for one feature layer.
    Determine the relative position grid of the centers, and the relative
    width and height.
    Arguments:
      feat_shape: Feature shape, used for computing relative position grids;
      size: Absolute reference sizes;
      ratios: Ratios to use on these features;
      img_shape: Image shape, used for computing height, width relatively to the
        former;
      offset: Grid offset.
    Return:
      y, x, h, w: Relative x and y grids, and height and width.
    """
    # Compute the position grid: simple way.
    # y, x = np.mgrid[0:feat_shape[0], 0:feat_shape[1]]
    # y = (y.astype(dtype) + offset) / feat_shape[0]
    # x = (x.astype(dtype) + offset) / feat_shape[1]
    # Weird SSD-Caffe computation using steps values...
    y, x = np.mgrid[0:feat_shape[0], 0:feat_shape[1]]
    y = (y.astype(dtype) + offset) * step / img_shape[0]
    x = (x.astype(dtype) + offset) * step / img_shape[1]

    # Expand dims to support easy broadcasting.
    y = np.expand_dims(y, axis=-1)
    x = np.expand_dims(x, axis=-1)

    # Compute relative height and width.
    # Tries to follow the original implementation of SSD for the order.
    num_anchors = len(sizes) + len(ratios)
    h = np.zeros((num_anchors, ), dtype=dtype)
    w = np.zeros((num_anchors, ), dtype=dtype)
    # Add first anchor boxes with ratio=1.
    h[0] = sizes[0] / img_shape[0]
    w[0] = sizes[0] / img_shape[1]
    di = 1
    if len(sizes) > 1:
        h[1] = math.sqrt(sizes[0] * sizes[1]) / img_shape[0]
        w[1] = math.sqrt(sizes[0] * sizes[1]) / img_shape[1]
        di += 1
    for i, r in enumerate(ratios):
        h[i+di] = sizes[0] / img_shape[0] / math.sqrt(r)
        w[i+di] = sizes[0] / img_shape[1] * math.sqrt(r)
    return y, x, h, w

def ssd_anchors_all_layers(img_shape,
                           layers_shape,
                           anchor_sizes,
                           anchor_ratios,
                           anchor_steps,
                           offset=0.5,
                           dtype=np.float32):
    """Compute anchor boxes for all feature layers.
    """
    layers_anchors = []
    for i, s in enumerate(layers_shape):
        anchor_bboxes = ssd_anchor_one_layer(img_shape, s,
                                             anchor_sizes[i],
                                             anchor_ratios[i],
                                             anchor_steps[i],
                                             offset=offset, dtype=dtype)
        layers_anchors.append(anchor_bboxes)
    return layers_anchors

def get_default_params():
    SSDParams = namedtuple('SSDParameters', ['img_shape',
                                         'num_classes',
                                         'no_annotation_label',
                                         'feat_layers',
                                         'feat_shapes',
                                         'anchor_size_bounds',
                                         'anchor_sizes',
                                         'anchor_ratios',
                                         'anchor_steps',
                                         'anchor_offset',
                                         'normalizations',
                                         'prior_scaling'
                                         ])
    default_params = SSDParams(
            img_shape=(300, 300),
            num_classes=21,
            no_annotation_label=21,
            feat_layers=['block4', 'block7', 'block8', 'block9', 'block10', 'block11'],
            feat_shapes=[(38, 38), (19, 19), (10, 10), (5, 5), (3, 3), (1, 1)],
            anchor_size_bounds=[0.15, 0.90],
            # anchor_size_bounds=[0.20, 0.90],
            anchor_sizes=[(21., 45.),
                          (45., 99.),
                          (99., 153.),
                          (153., 207.),
                          (207., 261.),
                          (261., 315.)],
            # anchor_sizes=[(30., 60.),
            #               (60., 111.),
            #               (111., 162.),
            #               (162., 213.),
            #               (213., 264.),
            #               (264., 315.)],
            anchor_ratios=[[2, .5],
                           [2, .5, 3, 1./3],
                           [2, .5, 3, 1./3],
                           [2, .5, 3, 1./3],
                           [2, .5],
                           [2, .5]],
            anchor_steps=[8, 16, 32, 64, 100, 300],
            anchor_offset=0.5,
            normalizations=[20, -1, -1, -1, -1, -1],
            prior_scaling=[0.1, 0.1, 0.2, 0.2]
            )
    return default_params

def get_anchors(config, dtype=np.float32):
    """Compute the default anchor boxes, given an image shape.
        """
    return ssd_anchors_all_layers(config.img_shape,
                                  config.feat_shapes,
                                  config.anchor_sizes,
                                  config.anchor_ratios,
                                  config.anchor_steps,
                                  config.anchor_offset,
                                  dtype)



NUM_CLASSES = 2
config = get_default_params()
config._replace(num_classes=NUM_CLASSES)
default_boxes =  get_anchors(config)