import tensorflow as tf


def compute_area(top_left, bot_right):
    """ Compute area given top_left and bottom_right coordinates
    Args:
        top_left: tensor (num_boxes, 2)
        bot_right: tensor (num_boxes, 2)
    Returns:
        area: tensor (num_boxes,)
    """
    # top_left: N x 2
    # bot_right: N x 2
    hw = tf.clip_by_value(bot_right - top_left, 0.0, 512.0)
    area = hw[..., 0] * hw[..., 1]

    return area


def compute_iou(boxes_a, boxes_b):
    """ Compute overlap between boxes_a and boxes_b
    Args:
        boxes_a: tensor (num_boxes_a, 4)
        boxes_b: tensor (num_boxes_b, 4)
    Returns:
        overlap: tensor (num_boxes_a, num_boxes_b)
    """
    # boxes_a => num_boxes_a, 1, 4
    boxes_a = tf.expand_dims(boxes_a, 1)

    # boxes_b => 1, num_boxes_b, 4
    boxes_b = tf.expand_dims(boxes_b, 0)
    top_left = tf.math.maximum(boxes_a[..., :2], boxes_b[..., :2])
    bot_right = tf.math.minimum(boxes_a[..., 2:], boxes_b[..., 2:])

    overlap_area = compute_area(top_left, bot_right)
    area_a = compute_area(boxes_a[..., :2], boxes_a[..., 2:])
    area_b = compute_area(boxes_b[..., :2], boxes_b[..., 2:])

    overlap = overlap_area / (area_a + area_b - overlap_area)

    return overlap


def compute_target(default_boxes, gt_boxes, gt_labels, iou_threshold=0.5):
    """ Compute regression and classification targets
    Args:
        default_boxes: tensor (num_default, 4)
                       of format (cx, cy, w, h)
        gt_boxes: tensor (num_gt, 4)
                  of format (xmin, ymin, xmax, ymax)
        gt_labels: tensor (num_gt,)
    Returns:
        gt_confs: classification targets, tensor (num_default,)
        gt_locs: regression targets, tensor (num_default, 4)
    """
    # print('gt_labels', gt_labels)
    # Convert default boxes to format (xmin, ymin, xmax, ymax)
    # in order to compute overlap with gt boxes
    if(gt_labels.shape[0] == 0):
        return tf.zeros(shape=[len(default_boxes)], dtype=tf.int64), default_boxes, tf.zeros(shape=[len(default_boxes)], dtype=tf.float32)
    transformed_default_boxes = transform_center_to_corner(default_boxes)
    iou = compute_iou(transformed_default_boxes, gt_boxes)

    best_gt_iou = tf.math.reduce_max(iou, 1)
    best_gt_idx = tf.math.argmax(iou, 1)

    best_default_iou = tf.math.reduce_max(iou, 0)
    best_default_idx = tf.math.argmax(iou, 0)

    best_gt_idx = tf.tensor_scatter_nd_update(
        best_gt_idx,
        tf.expand_dims(best_default_idx, 1),
        tf.range(best_default_idx.shape[0], dtype=tf.int64))

    # Normal way: use a for loop
    # for gt_idx, default_idx in enumerate(best_default_idx):
    #     best_gt_idx = tf.tensor_scatter_nd_update(
    #         best_gt_idx,
    #         tf.expand_dims([default_idx], 1),
    #         [gt_idx])

    best_gt_iou = tf.tensor_scatter_nd_update(
        best_gt_iou,
        tf.expand_dims(best_default_idx, 1),
        tf.ones_like(best_default_idx, dtype=tf.float32))

    gt_confs = tf.gather(gt_labels, best_gt_idx)
    gt_confs = tf.where(
        tf.less(best_gt_iou, iou_threshold),
        tf.zeros_like(gt_confs),
        gt_confs)
    gt_scores = tf.where(
        tf.less(best_gt_iou, iou_threshold),
        tf.zeros_like(best_gt_iou),
        best_gt_iou)

    gt_boxes = tf.gather(gt_boxes, best_gt_idx)
    gt_locs = encode(default_boxes, gt_boxes)
    return gt_confs, gt_locs, gt_scores




# def compute_target(default_boxes, gt_boxes, gt_labels, iou_threshold=0.5):
#     """ Compute regression and classification targets
#     Args:
#         default_boxes: tensor (num_default, 4)
#                        of format (cx, cy, w, h)
#         gt_boxes: tensor (num_gt, 4)
#                   of format (xmin, ymin, xmax, ymax)
#         gt_labels: tensor (num_gt,)
#     Returns:
#         gt_confs: classification targets, tensor (num_default,)
#         gt_locs: regression targets, tensor (num_default, 4)
#     """
#     # print('gt_labels', gt_labels)
#     # Convert default boxes to format (xmin, ymin, xmax, ymax)
#     # in order to compute overlap with gt boxes
#     if(gt_labels.shape[0] == 0):
#         return tf.zeros(shape=[len(default_boxes)], dtype=tf.int64), default_boxes
#     transformed_default_boxes = transform_center_to_corner(default_boxes)
#     iou = compute_iou(transformed_default_boxes, gt_boxes)

#     best_gt_iou = tf.math.reduce_max(iou, 1)
#     best_gt_idx = tf.math.argmax(iou, 1)

#     best_default_iou = tf.math.reduce_max(iou, 0)
#     best_default_idx = tf.math.argmax(iou, 0)

#     best_gt_idx = tf.tensor_scatter_nd_update(
#         best_gt_idx,
#         tf.expand_dims(best_default_idx, 1),
#         tf.range(best_default_idx.shape[0], dtype=tf.int64))

#     # Normal way: use a for loop
#     # for gt_idx, default_idx in enumerate(best_default_idx):
#     #     best_gt_idx = tf.tensor_scatter_nd_update(
#     #         best_gt_idx,
#     #         tf.expand_dims([default_idx], 1),
#     #         [gt_idx])

#     best_gt_iou = tf.tensor_scatter_nd_update(
#         best_gt_iou,
#         tf.expand_dims(best_default_idx, 1),
#         tf.ones_like(best_default_idx, dtype=tf.float32))

#     gt_confs = tf.gather(gt_labels, best_gt_idx)
#     gt_confs = tf.where(
#         tf.less(best_gt_iou, iou_threshold),
#         tf.zeros_like(gt_confs),
#         gt_confs)
#     gt_scores = tf.where(
#         tf.less(best_gt_iou, iou_threshold),
#         tf.zeros_like(best_gt_iou),
#         best_gt_iou)

#     gt_boxes = tf.gather(gt_boxes, best_gt_idx)
#     gt_locs = encode(default_boxes, gt_boxes)
#     return gt_confs, gt_locs, gt_scores


def encode(default_boxes, boxes, variance=[0.1, 0.2]):
    """ Compute regression values
    Args:
        default_boxes: tensor (num_default, 4)
                       of format (cx, cy, w, h)
        boxes: tensor (num_default, 4)
               of format (xmin, ymin, xmax, ymax)
        variance: variance for center point and size
    Returns:
        locs: regression values, tensor (num_default, 4)
    """
    # Convert boxes to (cx, cy, w, h) format
    transformed_boxes = transform_corner_to_center(boxes)

    locs = tf.concat([
        (transformed_boxes[..., :2] - default_boxes[:, :2]
         ) / (default_boxes[:, 2:] * variance[0]),
          tf.math.log(transformed_boxes[..., 2:] / default_boxes[:, 2:]) / variance[1]],
        axis=-1)

    return locs


def decode(default_boxes, locs, variance=[0.1, 0.2]):
    """ Decode regression values back to coordinates
    Args:
        default_boxes: tensor (num_default, 4)
                       of format (cx, cy, w, h)
        locs: tensor (batch_size, num_default, 4)
              of format (cx, cy, w, h)
        variance: variance for center point and size
    Returns:
        boxes: tensor (num_default, 4)
               of format (xmin, ymin, xmax, ymax)
    """
    locs = tf.concat([
        locs[..., :2] * variance[0] *
        default_boxes[:, 2:] + default_boxes[:, :2],
        tf.math.exp(locs[..., 2:] * variance[1]) * default_boxes[:, 2:]], axis=-1)
    
    boxes = transform_center_to_corner(locs)

    return boxes


def transform_corner_to_center(boxes):
    """ Transform boxes of format (xmin, ymin, xmax, ymax)
        to format (cx, cy, w, h)
    Args:
        boxes: tensor (num_boxes, 4)
               of format (xmin, ymin, xmax, ymax)
    Returns:
        boxes: tensor (num_boxes, 4)
               of format (cx, cy, w, h)
    """
    center_box = tf.concat([
        (boxes[..., :2] + boxes[..., 2:]) / 2,
        boxes[..., 2:] - boxes[..., :2]], axis=-1)

    return center_box


def transform_center_to_corner(boxes):
    """ Transform boxes of format (cx, cy, w, h)
        to format (xmin, ymin, xmax, ymax)
    Args:
        boxes: tensor (num_boxes, 4)
               of format (cx, cy, w, h)
    Returns:
        boxes: tensor (num_boxes, 4)
               of format (xmin, ymin, xmax, ymax)
    """
    corner_box = tf.concat([
        boxes[..., :2] - boxes[..., 2:] / 2,
        boxes[..., :2] + boxes[..., 2:] / 2], axis=-1)

    return corner_box


def compute_nms(boxes, scores, nms_threshold, limit=200):
    """ Perform Non Maximum Suppression algorithm
        to eliminate boxes with high overlap

    Args:
        boxes: tensor (num_boxes, 4)
               of format (xmin, ymin, xmax, ymax)
        scores: tensor (num_boxes,)
        nms_threshold: NMS threshold
        limit: maximum number of boxes to keep

    Returns:
        idx: indices of kept boxes
    """
    if boxes.shape[0] == 0:
        return tf.constant([], dtype=tf.int32)
    selected = [0]
    idx = tf.argsort(scores, direction='DESCENDING')
    idx = idx[:limit]
    boxes = tf.gather(boxes, idx)

    iou = compute_iou(boxes, boxes)

    while True:
        row = iou[selected[-1]]
        next_indices = row <= nms_threshold
        # iou[:, ~next_indices] = 1.0
        iou = tf.where(
            tf.expand_dims(tf.math.logical_not(next_indices), 0),
            tf.ones_like(iou, dtype=tf.float32),
            iou)

        if not tf.math.reduce_any(next_indices):
            break

        selected.append(tf.argsort(
            tf.dtypes.cast(next_indices, tf.int32), direction='DESCENDING')[0].numpy())

    return tf.gather(idx, selected)

def reshape_list(l, shape=None):
    """Reshape list of (list): 1D to 2D or the other way around.
    Args:
      l: List or List of list.
      shape: 1D or 2D shape.
    Return
      Reshaped list.
    """
    r = []
    if shape is None:
        # Flatten everything.
        for a in l:
            if isinstance(a, (list, tuple)):
                r = r + list(a)
            else:
                r.append(a)
    else:
        # Reshape to list of list.
        i = 0
        for s in shape:
            if s == 1:
                r.append(l[i])
            else:
                r.append(l[i:i+s])
            i += s
    return r


def tf_ssd_bboxes_encode(labels,
                         bboxes,
                         anchors,
                         num_classes,
                         no_annotation_label,
                         ignore_threshold=0.5,
                         prior_scaling=[0.1, 0.1, 0.2, 0.2],
                         dtype=tf.float32,
                         scope='ssd_bboxes_encode'):
    """Encode groundtruth labels and bounding boxes using SSD net anchors.
    Encoding boxes for all feature layers.
    Arguments:
      labels: 1D Tensor(int64) containing groundtruth labels;
      bboxes: Nx4 Tensor(float) with bboxes relative coordinates;
      anchors: List of Numpy array with layer anchors;
      matching_threshold: Threshold for positive match with groundtruth bboxes;
      prior_scaling: Scaling of encoded coordinates.
    Return:
      (target_labels, target_localizations, target_scores):
        Each element is a list of target Tensors.
    """
    with tf.name_scope(scope):
        target_labels = []
        target_localizations = []
        target_scores = []
        for i, anchors_layer in enumerate(anchors):
            with tf.name_scope('bboxes_encode_block_%i' % i):
                t_labels, t_loc, t_scores = \
                    tf_ssd_bboxes_encode_layer(labels, bboxes, anchors_layer,
                                               num_classes, no_annotation_label,
                                               ignore_threshold,
                                               prior_scaling, dtype)
                t_labels = tf.reshape(t_labels, (-1))
                t_loc = tf.reshape(t_loc, (-1, 4))
                t_scores = tf.reshape(t_scores, (-1))
                target_labels.append(t_labels)
                target_localizations.append(t_loc)
                target_scores.append(t_scores)
                                
        target_labels = tf.concat(target_labels, axis=0)   
        target_localizations = tf.concat(target_localizations, axis=0)  
        target_scores = tf.concat(target_scores, axis=0)        
        return target_labels, target_localizations, target_scores


def tf_ssd_bboxes_encode_layer(labels,
                               bboxes,
                               anchors_layer,
                               num_classes,
                               no_annotation_label,
                               ignore_threshold=0.5,
                               prior_scaling=[0.1, 0.1, 0.2, 0.2],
                               dtype=tf.float32):
    """Encode groundtruth labels and bounding boxes using SSD anchors from
    one layer.
    Arguments:
      labels: 1D Tensor(int64) containing groundtruth labels;
      bboxes: Nx4 Tensor(float) with bboxes relative coordinates;
      anchors_layer: Numpy array with layer anchors;
      matching_threshold: Threshold for positive match with groundtruth bboxes;
      prior_scaling: Scaling of encoded coordinates.
    Return:
      (target_labels, target_localizations, target_scores): Target Tensors.
    """
    # Anchors coordinates and volume.
    yref, xref, href, wref = anchors_layer
    ymin = yref - href / 2.
    xmin = xref - wref / 2.
    ymax = yref + href / 2.
    xmax = xref + wref / 2.
    vol_anchors = (xmax - xmin) * (ymax - ymin)

    # Initialize tensors...
    shape = (yref.shape[0], yref.shape[1], href.size)
    feat_labels = tf.zeros(shape, dtype=tf.int64)
    feat_scores = tf.zeros(shape, dtype=dtype)

    feat_ymin = tf.zeros(shape, dtype=dtype)
    feat_xmin = tf.zeros(shape, dtype=dtype)
    feat_ymax = tf.ones(shape, dtype=dtype)
    feat_xmax = tf.ones(shape, dtype=dtype)

    def jaccard_with_anchors(bbox):
        """Compute jaccard score between a box and the anchors.
        """
        int_ymin = tf.maximum(ymin, bbox[0])
        int_xmin = tf.maximum(xmin, bbox[1])
        int_ymax = tf.minimum(ymax, bbox[2])
        int_xmax = tf.minimum(xmax, bbox[3])
        h = tf.maximum(int_ymax - int_ymin, 0.)
        w = tf.maximum(int_xmax - int_xmin, 0.)
        # Volumes.
        inter_vol = h * w
        union_vol = vol_anchors - inter_vol \
            + (bbox[2] - bbox[0]) * (bbox[3] - bbox[1])
        jaccard = tf.math.divide(inter_vol, union_vol)
        return jaccard

    def intersection_with_anchors(bbox):
        """Compute intersection between score a box and the anchors.
        """
        int_ymin = tf.maximum(ymin, bbox[0])
        int_xmin = tf.maximum(xmin, bbox[1])
        int_ymax = tf.minimum(ymax, bbox[2])
        int_xmax = tf.minimum(xmax, bbox[3])
        h = tf.maximum(int_ymax - int_ymin, 0.)
        w = tf.maximum(int_xmax - int_xmin, 0.)
        inter_vol = h * w
        scores = tf.math.divide(inter_vol, vol_anchors)
        return scores

    def condition(i, feat_labels, feat_scores,
                  feat_ymin, feat_xmin, feat_ymax, feat_xmax):
        """Condition: check label index.
        """
        r = tf.less(i, tf.shape(labels))
        return r[0]

    def body(i, feat_labels, feat_scores,
             feat_ymin, feat_xmin, feat_ymax, feat_xmax):
        """Body: update feature labels, scores and bboxes.
        Follow the original SSD paper for that purpose:
          - assign values when jaccard > 0.5;
          - only update if beat the score of other bboxes.
        """
        # Jaccard score.
        label = labels[i]
        bbox = bboxes[i]
        jaccard = jaccard_with_anchors(bbox)
        # Mask: check threshold + scores + no annotations + num_classes.
        mask = tf.greater(jaccard, feat_scores)
        # mask = tf.logical_and(mask, tf.greater(jaccard, matching_threshold))
        mask = tf.logical_and(mask, feat_scores > -0.5)
        mask = tf.logical_and(mask, label < num_classes)
        imask = tf.cast(mask, tf.int64)
        fmask = tf.cast(mask, dtype)
        # Update values using mask.
        feat_labels = imask * label + (1 - imask) * feat_labels
        feat_scores = tf.where(mask, jaccard, feat_scores)

        feat_ymin = fmask * bbox[0] + (1 - fmask) * feat_ymin
        feat_xmin = fmask * bbox[1] + (1 - fmask) * feat_xmin
        feat_ymax = fmask * bbox[2] + (1 - fmask) * feat_ymax
        feat_xmax = fmask * bbox[3] + (1 - fmask) * feat_xmax

        # Check no annotation label: ignore these anchors...
        # interscts = intersection_with_anchors(bbox)
        # mask = tf.logical_and(interscts > ignore_threshold,
        #                       label == no_annotation_label)
        # # Replace scores by -1.
        # feat_scores = tf.where(mask, -tf.cast(mask, dtype), feat_scores)

        return [i+1, feat_labels, feat_scores,
                feat_ymin, feat_xmin, feat_ymax, feat_xmax]
    # Main loop definition.
    i = 0
    [i, feat_labels, feat_scores,
     feat_ymin, feat_xmin,
     feat_ymax, feat_xmax] = tf.while_loop(condition, body,
                                           [i, feat_labels, feat_scores,
                                            feat_ymin, feat_xmin,
                                            feat_ymax, feat_xmax])
    # Transform to center / size.
    feat_cy = (feat_ymax + feat_ymin) / 2.
    feat_cx = (feat_xmax + feat_xmin) / 2.
    feat_h = feat_ymax - feat_ymin
    feat_w = feat_xmax - feat_xmin
    # Encode features.
    feat_cy = (feat_cy - yref) / href / prior_scaling[0]
    feat_cx = (feat_cx - xref) / wref / prior_scaling[1]
    feat_h = tf.math.log(feat_h / href) / prior_scaling[2]
    feat_w = tf.math.log(feat_w / wref) / prior_scaling[3]
    # Use SSD ordering: x / y / w / h instead of ours.
    feat_localizations = tf.stack([feat_cx, feat_cy, feat_w, feat_h], axis=-1)
    return feat_labels, feat_localizations, feat_scores

