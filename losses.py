import tensorflow as tf
import numpy as np

def hard_negative_mining(loss, gt_confs, neg_ratio):
    """ Hard negative mining algorithm
        to pick up negative examples for back-propagation
        base on classification loss values
    Args:
        loss: list of classification losses of all default boxes (B, num_default)
        gt_confs: classification targets (B, num_default)
        neg_ratio: negative / positive ratio
    Returns:
        conf_loss: classification loss
        loc_loss: regression loss
    """
    # loss: B x N
    # gt_confs: B x N
    pos_idx = gt_confs > 0
    num_pos = tf.reduce_sum(tf.dtypes.cast(pos_idx, tf.int32), axis=1)
    num_neg = num_pos * neg_ratio
    print("numneg = ", num_neg.numpy())
    print("num_pos = ", num_pos.numpy())
    rank = tf.argsort(loss, axis=1, direction='DESCENDING')
    # print(rank.shape)
    rank = tf.argsort(rank, axis=1)
    # print(rank.shape)
    neg_idx = rank < tf.expand_dims(num_neg, 1)
    # print(neg_idx.shape)
    return pos_idx, neg_idx


class SSDLosses(object):
    """ Class for SSD Losses
    Attributes:
        neg_ratio: negative / positive ratio
        num_classes: number of classes
    """

    def __init__(self, neg_ratio, num_classes):
        self.neg_ratio = neg_ratio
        self.num_classes = num_classes

    def __call__(self, confs, locs, gt_confs, gt_locs):
        """ Compute losses for SSD
            regression loss: smooth L1
            classification loss: cross entropy
        Args:
            confs: outputs of classification heads (B, num_default, num_classes)
            locs: outputs of regression heads (B, num_default, 4)
            gt_confs: classification targets (B, num_default)
            gt_locs: regression targets (B, num_default, 4)
        Returns:
            conf_loss: classification loss
            loc_loss: regression loss
        """
        # print(confs.shape, locs.shape, gt_confs.shape, gt_locs.shape, '------------')
        cross_entropy = tf.keras.losses.SparseCategoricalCrossentropy(
            from_logits=True, reduction='none')
        
        # compute classification losses
        # without reduction
        temp_loss = cross_entropy(
            gt_confs, confs)
        pos_idx, neg_idx = hard_negative_mining(
            temp_loss, gt_confs, self.neg_ratio)
        
        # classification loss will consist of positive and negative examples
        cross_entropy = tf.keras.losses.SparseCategoricalCrossentropy(
            from_logits=True, reduction='sum')
        smooth_l1_loss = tf.keras.losses.Huber(reduction='sum')
        
        conf_loss = tf.cond(tf.greater(pos_idx.shape[0], 0), lambda: cross_entropy(
            gt_confs[tf.math.logical_or(pos_idx, neg_idx)],
            confs[tf.math.logical_or(pos_idx, neg_idx)]), lambda: lambda : tf.constant(0.0, dtype=tf.float32))
        
        # regression loss only consist of positive examples
        loc_loss = tf.cond(tf.greater(pos_idx.shape[0], 0),lambda: smooth_l1_loss(
            gt_locs[pos_idx],
            locs[pos_idx]), lambda : tf.constant(0.0, dtype=tf.float32))

        num_pos = tf.reduce_sum(tf.dtypes.cast(pos_idx, tf.float32))
        # print("0", loc_loss.numpy(), gt_locs[pos_idx].numpy(), pos_idx.numpy())
        tmp = tf.constant(0.0, shape=loc_loss.shape, dtype=tf.float32)
        conf_loss = tf.cond(tf.equal(num_pos, 0),lambda: tmp,lambda: conf_loss / num_pos)
        loc_loss = tf.cond(tf.equal(num_pos, 0),lambda: tmp,lambda: loc_loss / num_pos) 
        # print("1", conf_loss.numpy(), loc_loss.numpy())
        return conf_loss, loc_loss

def ssd_losses(logits, localisations,
               gclasses, glocalisations, gscores,
               match_threshold=0.5,
               negative_ratio=3.,
               alpha=1.,
               label_smoothing=0.):
    """
    Attributes:
        logits =    
    """
    lshape = logits[0].shape[:5]
    num_classes = lshape[-1]
    batch_size = lshape[0]

    # Flatten out all vectors!
    flogits = []
    fgclasses = []
    fgscores = []
    flocalisations = []
    fglocalisations = []
    for i in range(len(logits)):
        flogits.append(tf.reshape(logits[i], [-1, num_classes]))
        fgclasses.append(tf.reshape(gclasses[i], [-1]))
        fgscores.append(tf.reshape(gscores[i], [-1]))
        flocalisations.append(tf.reshape(localisations[i], [-1, 4]))
        fglocalisations.append(tf.reshape(glocalisations[i], [-1, 4]))
    # And concat the crap!
    logits = tf.concat(flogits, axis=0)
    gclasses = tf.concat(fgclasses, axis=0)
    gscores = tf.concat(fgscores, axis=0)
    localisations = tf.concat(flocalisations, axis=0)
    glocalisations = tf.concat(fglocalisations, axis=0)
    dtype = logits.dtype

    # Compute positive matching mask...
    pmask = gscores > match_threshold
    fpmask = tf.cast(pmask, dtype)
    n_positives = tf.reduce_sum(fpmask)

    # Hard negative mining...
    no_classes = tf.cast(pmask, tf.int32)
    predictions = tf.nn.softmax(logits)
    nmask = tf.logical_and(tf.logical_not(pmask),
                           gscores > -0.5)
    fnmask = tf.cast(nmask, dtype)
    nvalues = tf.where(nmask,
                       predictions[:, 0],
                       1. - fnmask)
    nvalues_flat = tf.reshape(nvalues, [-1])
    # Number of negative entries to select.
    max_neg_entries = tf.cast(tf.reduce_sum(fnmask), tf.int32)
    n_neg = tf.cast(negative_ratio * n_positives, tf.int32) + batch_size
    n_neg = tf.minimum(n_neg, max_neg_entries)

    val, idxes = tf.nn.top_k(-nvalues_flat, k=n_neg)
    max_hard_pred = -val[-1]
    # Final negative mask.
    nmask = tf.logical_and(nmask, nvalues < max_hard_pred)
    fnmask = tf.cast(nmask, dtype)

    
    lossp = tf.nn.sparse_softmax_cross_entropy_with_logits(logits=logits,
                                                          labels=gclasses)
    lossp = tf.math.divide(tf.reduce_sum(lossp * fpmask), batch_size)
    

    
    lossn = tf.nn.sparse_softmax_cross_entropy_with_logits(logits=logits,
                                                          labels=no_classes)
    lossn = tf.math.divide(tf.reduce_sum(lossn * fnmask), batch_size)
        

    # Add localization loss: smooth L1, L2, ...
    
    # Weights Tensor: positive mask + random negative.
    weights = tf.expand_dims(alpha * fpmask, axis=-1)
    loss_loc = abs_smooth(localisations - glocalisations)
    loss_loc = tf.where(tf.math.equal(loss_loc,tf.constant(np.inf)), tf.constant(0.0), loss_loc)
    loss_loc = tf.where(tf.math.equal(loss_loc,tf.constant(np.nan)), tf.constant(0.0), loss_loc)
    loss_loc = tf.math.divide(tf.reduce_sum(loss_loc * weights), batch_size)
    loss_conf  = lossp + lossn
    # loss = loss_conf + loss_loc
    # tf.keras.clip(
    
    return loss_conf, loss_loc

def abs_smooth(x):
    """Smoothed absolute function. Useful to compute an L1 smooth error.
    Define as:
        x^2 / 2         if abs(x) < 1
        abs(x) - 0.5    if abs(x) > 1
    We use here a differentiable definition using min(x) and abs(x). Clearly
    not optimal, but good enough for our purpose!
    """
    absx = tf.abs(x)
    minx = tf.minimum(absx, 1)
    r = 0.5 * ((absx - 1) * minx + absx)
    return r


def create_losses(neg_ratio, num_classes):
    criterion = SSDLosses(neg_ratio, num_classes)

    return criterion




