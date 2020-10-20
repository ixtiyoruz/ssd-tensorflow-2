import argparse
import tensorflow as tf
import os
import sys
import time
import yaml
import numpy as np
from tensorflow.keras.optimizers.schedules import PiecewiseConstantDecay
# from voc_data import create_batch_generator
from datasets.COCOdataset import COCOdataset
from anchor import generate_default_boxes
from network_new import create_ssd
from losses import create_losses


parser = argparse.ArgumentParser()
parser.add_argument('--data-dir', default='../dataset')
parser.add_argument('--data-year', default='2017')
parser.add_argument('--arch', default='ssd300')
parser.add_argument('--batch-size', default=10, type=int)
parser.add_argument('--num-batches', default=-1, type=int)
parser.add_argument('--neg-ratio', default=3, type=int)
parser.add_argument('--initial-lr', default=1e-3, type=float)
parser.add_argument('--momentum', default=0.9, type=float)
parser.add_argument('--weight-decay', default=5e-4, type=float)
parser.add_argument('--num-epochs', default=120, type=int)
parser.add_argument('--checkpoint-dir', default='checkpoints')
parser.add_argument('--pretrained-type', default='base')
parser.add_argument('--gpu-id', default='0')

args = parser.parse_args()

os.environ['CUDA_VISIBLE_DEVICES'] = args.gpu_id

NUM_CLASSES = 2


@tf.function
def train_step(imgs, gt_confs, gt_locs, ssd, criterion, optimizer):
    with tf.GradientTape() as tape:
        out = ssd(imgs)
        confs_, locs_ = out[:6], out[6:]
        print(type(confs_[0]))
        confs  = []
        locs = []
        for i in range(len(confs_)):            
            confs.append(tf.reshape(confs_[i], [confs_[i].shape[0], -1, NUM_CLASSES]))
            locs.append(tf.reshape(locs_[i], [locs_[i].shape[0], -1, 4]))        
        confs = tf.concat(confs, axis=1)
        locs = tf.concat(locs, axis=1)
        conf_loss, loc_loss = criterion(
            confs, locs, gt_confs, gt_locs)

        loss = conf_loss + loc_loss
        l2_loss = [tf.nn.l2_loss(t) for t in ssd.trainable_variables]
        l2_loss = args.weight_decay * tf.math.reduce_sum(l2_loss)
        loss += l2_loss

    gradients = tape.gradient(loss, ssd.trainable_variables)
    optimizer.apply_gradients(zip(gradients, ssd.trainable_variables))

    return loss, conf_loss, loc_loss, l2_loss


if __name__ == '__main__':
    os.makedirs(args.checkpoint_dir, exist_ok=True)

    with open('./config.yml') as f:
        cfg = yaml.load(f)

    try:
        config = cfg['SSD300']#[args.arch.upper()]
    except AttributeError:
        raise ValueError('Unknown architecture: {}'.format(args.arch))

    default_boxes = generate_default_boxes(config)

    # batch_generator, val_generator, info = create_batch_generator(
    #     args.data_dir, args.data_year, default_boxes,
    #     config['image_size'],
    #     args.batch_size, args.num_batches,
    #     mode='train', augmentation=['flip'])  # the patching algorithm is currently causing bottleneck sometimes
    # batch_generator, val_generator, info = create_batch_generator(
         # args.data_year, default_boxes,
        # config['image_size'],
        # args.batch_size, args.num_batches,
        # mode='train', augmentation=['flip'])  # the patching algorithm is currently causing bottleneck sometimes
    voc = COCOdataset('/media/essys/bd21e577-e5db-47a3-8845-f27be3f083a4/dataset/coco',default_boxes,batch_size=args.batch_size,datatype='train2017',
                     input_shape=config['image_size'], augmentation=['flip'])
    voc_val = COCOdataset('/media/essys/bd21e577-e5db-47a3-8845-f27be3f083a4/dataset/coco',default_boxes,datatype='val2017',
                     input_shape=config['image_size'], augmentation=['flip'])
    batch_generator= voc.generate()
    val_generator= voc_val.generate()
    try:
        ssd = create_ssd(NUM_CLASSES, args.arch,
                        args.pretrained_type,
                        checkpoint_dir=args.checkpoint_dir)        
    except Exception as e:
        print(e)
        print('The program is exiting...')
        sys.exit()

    criterion = create_losses(args.neg_ratio, NUM_CLASSES)

    steps_per_epoch = len(voc) // args.batch_size

    lr_fn = PiecewiseConstantDecay(
        boundaries=[int(steps_per_epoch * args.num_epochs * 2 / 3),
                    int(steps_per_epoch * args.num_epochs * 5 / 6)],
        values=[args.initial_lr, args.initial_lr * 0.1, args.initial_lr * 0.01])
    
    optimizer = tf.keras.optimizers.SGD(
        learning_rate=lr_fn,
        momentum=args.momentum)

    train_log_dir = 'logs/train'
    val_log_dir = 'logs/val'
    train_summary_writer = tf.summary.create_file_writer(train_log_dir)
    val_summary_writer = tf.summary.create_file_writer(val_log_dir)

    for epoch in range(args.num_epochs):
        avg_loss = 0.0
        avg_conf_loss = 0.0
        avg_loc_loss = 0.0
        start = time.time()
        for i, (imgs, gt_confs, gt_locs) in enumerate(batch_generator):            
            # print(imgs.shape, gt_confs.shape, gt_locs.shape)
            loss, conf_loss, loc_loss, l2_loss = train_step(
                imgs, gt_confs, gt_locs, ssd, criterion, optimizer)
            
            avg_loss = (avg_loss * i + loss.numpy()) / (i + 1)
            avg_conf_loss = (avg_conf_loss * i + conf_loss.numpy()) / (i + 1)
            avg_loc_loss = (avg_loc_loss * i + loc_loss.numpy()) / (i + 1)
            if(np.any(np.isnan(loc_loss.numpy()))):
                break
            if (i + 1) % 50 == 0:
                print('Epoch: {} Batch {} Time: {:.2}s | Loss: {:.4f} Conf: {:.4f} Loc: {:.4f}'.format(
                    epoch + 1, i + 1, time.time() - start, avg_loss, avg_conf_loss, avg_loc_loss))
            if((i+1) % 200 ==0):
                ssd.save_weights(os.path.join(args.checkpoint_dir, 'ssd_epoch_{}_step_{}.h5'.format(epoch + 1, i+1)))
        avg_val_loss = 0.0
        avg_val_conf_loss = 0.0
        avg_val_loc_loss = 0.0
        for i, (_, imgs, gt_confs, gt_locs) in enumerate(val_generator):
            val_confs, val_locs = ssd(imgs)
            val_conf_loss, val_loc_loss = criterion(
                val_confs, val_locs, gt_confs, gt_locs)
            val_loss = val_conf_loss + val_loc_loss
            avg_val_loss = (avg_val_loss * i + val_loss.numpy()) / (i + 1)
            avg_val_conf_loss = (avg_val_conf_loss * i + val_conf_loss.numpy()) / (i + 1)
            avg_val_loc_loss = (avg_val_loc_loss * i + val_loc_loss.numpy()) / (i + 1)

        with train_summary_writer.as_default():
            tf.summary.scalar('loss', avg_loss, step=epoch)
            tf.summary.scalar('conf_loss', avg_conf_loss, step=epoch)
            tf.summary.scalar('loc_loss', avg_loc_loss, step=epoch)

        with val_summary_writer.as_default():
            tf.summary.scalar('loss', avg_val_loss, step=epoch)
            tf.summary.scalar('conf_loss', avg_val_conf_loss, step=epoch)
            tf.summary.scalar('loc_loss', avg_val_loc_loss, step=epoch)

        
        ssd.save_weights(
            os.path.join(args.checkpoint_dir, 'ssd_epoch_{}.h5'.format(epoch + 1)))
