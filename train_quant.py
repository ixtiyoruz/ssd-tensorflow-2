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
parser.add_argument('--arch', default='ssd300')
parser.add_argument('--batch-size', default=16, type=int)
parser.add_argument('--num-batches', default=100, type=int)
parser.add_argument('--neg-ratio', default=3, type=int)
parser.add_argument('--initial-lr', default=7e-4, type=float)
parser.add_argument('--momentum', default=0.9, type=float)
parser.add_argument('--weight-decay', default=1e-5, type=float)
parser.add_argument('--num-epochs', default=120, type=int)
parser.add_argument('--checkpoint-dir', default='checkpoints')
parser.add_argument('--pretrained-type', default='base')
parser.add_argument('--gpu-id', default='2,3')
parser.add_argument('--freeze', default=1, type=int)
parser.add_argument('--quant', default=0, type=int)

args = parser.parse_args()
print(tf.executing_eagerly())
print("running gpus :" , args.gpu_id)
os.environ['CUDA_VISIBLE_DEVICES'] = args.gpu_id

NUM_CLASSES = 2
def post_process(confs, locs,scores, default_boxes, mode=1):
    # confs = tf.squeeze(confs, 0)
    # locs = tf.squeeze(locs, 0)
    # i have o return the locs back
    # print(scores)
    # print(confs)
    # print(locs)
    print(tf.math.reduce_max(locs), tf.math.reduce_min(locs))
    newres = decode(default_boxes, locs).numpy()
    if(mode == 2):
        confs = tf.math.softmax(confs, axis=-1)
        classes = tf.math.argmax(confs, axis=-1)
        scores = tf.math.reduce_max(confs, axis=-1)
            
    out_boxes = []
    out_labels = []
    out_scores = []
    # print(confs.shape,classes.shape, scores.shape, boxes.shape)
    for c in range(1, NUM_CLASSES):
        if(mode == 1):
            cls_scores = np.zeros(np.shape(confs))
            cls_scores[confs  == c]  = confs[confs  == c]
        else:
            cls_scores = confs[:, c]

        score_idx = cls_scores > 0.5
        # cls_boxes = tf.boolean_mask(boxes, score_idx)
        # cls_scores = tf.boolean_mask(cls_scores, score_idx)
        cls_boxes = newres[score_idx]
        cls_scores = cls_scores[score_idx]

        nms_idx = compute_nms(cls_boxes, cls_scores, 0.35, 200)
        cls_boxes = tf.gather(cls_boxes, nms_idx)
        cls_scores = tf.gather(cls_scores, nms_idx)
        cls_labels = [c] * cls_boxes.shape[0]

        out_boxes.append(cls_boxes)
        out_labels.extend(cls_labels)
        out_scores.append(cls_scores)

    out_boxes = tf.concat(out_boxes, axis=0)
    out_scores = tf.concat(out_scores, axis=0)

    boxes = tf.clip_by_value(out_boxes, 0.0, 1.0).numpy()
    classes = np.array(out_labels)
    scores = out_scores.numpy()

    return boxes, classes, scores
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
      
@tf.function
def train_step(imgs, gt_confs, gt_locs,gt_scores, ssd, criterion, optimizer):
    with tf.GradientTape() as tape:
        # start = time.time()
        out = ssd(imgs)
        # print(time.time() - start, " s spent for model out")
        confs_, locs_ = out[:6], out[6:]
        
        confs  = []
        locs = []
        for i in range(len(confs_)):            
            confs.append(tf.reshape(confs_[i], [confs_[i].shape[0], -1, NUM_CLASSES]))
            locs.append(tf.reshape(locs_[i], [locs_[i].shape[0], -1, 4]))     
        confs = tf.concat(confs, axis=1)
        locs = tf.concat(locs, axis=1)
        # print(confs.shape, locs.shape)
        # start = time.time()
        conf_loss, loc_loss = criterion(
            confs, locs, gt_confs, gt_locs, gt_scores)
        # print(time.time() - start, " s spent for loss calculation")
        loss = conf_loss + loc_loss
        l2_loss = [tf.nn.l2_loss(t) for t in ssd.trainable_variables]
        l2_loss = args.weight_decay * tf.math.reduce_sum(l2_loss)
        loss += l2_loss

    gradients = tape.gradient(loss, ssd.trainable_variables)
    optimizer.apply_gradients(zip(gradients, ssd.trainable_variables))

    return loss, conf_loss, loc_loss, confs, locs
# @tf.function
# def distributed_train_step(imgs, gt_confs, gt_locs, q_aware_model, criterion, optimize, strategy):
#   loss, conf_loss, loc_loss, l2_loss,confs_pred,locs_pred = strategy.run(train_step, args=(imgs, gt_confs, gt_locs, q_aware_model, criterion, optimize))
#   print(conf_loss.values[0].shape, '-------------')
#   return strategy.reduce(tf.distribute.ReduceOp.SUM, loss, axis=None), conf_loss.values[0], loc_loss.values[0], l2_loss,confs_pred.values[0],locs_pred.values[0]
# @tf.function
# def distributed_test_step(imgs, gt_confs, gt_locs, q_aware_model, criterion, optimize, strategy):
#   return strategy.run(train_step, args=(imgs, gt_confs, gt_locs, q_aware_model, criterion, optimize))


if __name__ == '__main__':
    os.makedirs(args.checkpoint_dir, exist_ok=True)

    with open('./config.yml') as f:
        cfg = yaml.load(f)

    try:
        config = cfg['SSD300']#[args.arch.upper()]
    except AttributeError:
        raise ValueError('Unknown architecture: {}'.format(args.arch))
    default_boxes = generate_default_boxes(config)
    # config = get_default_params()
    # config._replace(num_classes=NUM_CLASSES)
    # default_boxes =  get_anchors(config)
    # batch_generator, val_generator, info 
    # voc = VOCDataset(
    #     args.data_dir, args.data_year, default_boxes,
    #     config['image_size'],
    #     batch_size=args.batch_size,
    #     augmentation=['flip'])  # the patching algorithm is currently causing bottleneck sometimes
    # batch_generator = voc.generate(subset='train',num=2000)
    # val_generator = voc.generate(subset='val',num=2)
    
    # mirrored_strategy = tf.distribute.MirroredStrategy()
    # mirrored_strategy = tf.distribute.MirroredStrategy(devices=["/gpu:0", "/gpu:1", "/gpu:2"])
    # GLOBAL_BATCH_SIZE = args.batch_size * 3
    voc = COCOdataset('/media/essys/bd21e577-e5db-47a3-8845-f27be3f083a4/dataset/coco',default_boxes,batch_size=args.batch_size,datatype='train2017',
                      input_shape=[config['image_size'], config['image_size']], augmentation=['flip'])
    voc_val = COCOdataset('/media/essys/bd21e577-e5db-47a3-8845-f27be3f083a4/dataset/coco',default_boxes,datatype='val2017',
                      input_shape=[config['image_size'], config['image_size']], augmentation=['none'])
    # batch_generator= voc.generate(20000)
    # val_generator= voc_val.generate(20000)

    try:
        ssd = create_ssd(NUM_CLASSES, args.arch,args.batch_size,
                        args.pretrained_type,
                        checkpoint_dir=args.checkpoint_dir,scope=None, freeze=args.freeze, quant=False)        
    except Exception as e:
        print(e)
        print('The program is exiting...')
        sys.exit()
    


    print('quantizing enabled : ', args.quant)
    print('freezing enabled : ', args.freeze)
    if(args.quant > 0):                
        q_aware_model = quantize_model(ssd)
    else:
        q_aware_model = ssd
    # criterion = create_losses(args.neg_ratio, NUM_CLASSES)
    criterion = ssd_losses

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
    global_step = 0
    for epoch in range(args.num_epochs):
        avg_loss = 0.0
        avg_conf_loss = 0.0
        avg_loc_loss = 0.0
        start = time.time()
        batch_generator= voc.generate(2000)
        val_generator= voc_val.generate(2)
        for i, (imgs, gt_confs, gt_locs,gt_scores, o_imgs, o_boxes, o_labels) in enumerate(batch_generator):  
            global_step = global_step + 1
            # print(imgs.shape, gt_confs.shape, gt_locs.shape)
            loss, conf_loss, loc_loss,confs_pred,locs_pred = train_step(
                imgs, gt_confs, gt_locs,gt_scores, q_aware_model, criterion, optimizer)
            avg_loss = (avg_loss * i + loss.numpy()) / (i + 1)
            avg_conf_loss = (avg_conf_loss * i + conf_loss.numpy()) / (i + 1)
            avg_loc_loss = (avg_loc_loss * i + loc_loss.numpy()) / (i + 1)
            if (i + 1) % 20 == 0:
                print('Epoch: {} Batch {} Time: {:.2}s | Loss: {:.4f} Conf: {:.4f} Loc: {:.4f}'.format(
                    epoch + 1, i + 1, time.time() - start, avg_loss, avg_conf_loss, avg_loc_loss))
                with train_summary_writer.as_default():
                    tf.summary.scalar('loss', avg_loss, step=global_step)
                    tf.summary.scalar('conf_loss', avg_conf_loss, step=global_step)
                    tf.summary.scalar('loc_loss', avg_loc_loss, step=global_step)
                    
                    # here we draw bounding boxes and print summary
                    gboxes, gclasses, gscores = post_process(gt_confs.numpy()[0], gt_locs.numpy()[0], gt_scores.numpy()[0], default_boxes)
                    boxes, classes, scores = post_process(confs_pred.numpy()[0], locs_pred.numpy()[0],None, default_boxes, mode=2)
                    img_gt = cv2.cvtColor(np.uint8(o_imgs[0]).copy(), cv2.COLOR_RGB2BGR)
                    img_pred = cv2.cvtColor(np.uint8(o_imgs[0]).copy(), cv2.COLOR_RGB2BGR)
                    draw_image(img_gt, gboxes, gclasses)
                    draw_image(img_pred, boxes ,classes, mode='pred')
                    tf.summary.image("train/gt", np.expand_dims(img_gt, 0),step=global_step)
                    tf.summary.image("train/pred", np.expand_dims(img_pred, 0),step=global_step)                    
                
            if((i+1) % 500 ==0):
                ssd.save_weights(os.path.join(args.checkpoint_dir, 'ssd_epoch_{}_step_{}.h5'.format(epoch + 1, i+1)))            
        avg_val_loss = 0.0
        avg_val_conf_loss = 0.0
        avg_val_loc_loss = 0.0
        for i, (imgs, gt_confs, gt_locs,gt_scores, o_imgs, o_boxes, o_labels) in enumerate(val_generator):
            out = ssd(imgs)
            val_confs_, val_locs_ = out[:6], out[6:]
            
            val_confs  = []
            val_locs = []
            for j in range(len(val_confs_)):            
                val_confs.append(tf.reshape(val_confs_[j], [val_confs_[j].shape[0], -1, NUM_CLASSES]))
                val_locs.append(tf.reshape(val_locs_[j], [val_locs_[j].shape[0], -1, 4]))        
            val_confs = tf.concat(val_confs, axis=1)
            val_locs = tf.concat(val_locs, axis=1)
            val_conf_loss, val_loc_loss = criterion(
                val_confs, val_locs, gt_confs, gt_locs,gt_scores)
            val_loss = val_conf_loss + val_loc_loss
            avg_val_loss = (avg_val_loss * i + val_loss.numpy()) / (i + 1)
            avg_val_conf_loss = (avg_val_conf_loss * i + val_conf_loss.numpy()) / (i + 1)
            avg_val_loc_loss = (avg_val_loc_loss * i + val_loc_loss.numpy()) / (i + 1)

        
        with val_summary_writer.as_default():
            tf.summary.scalar('loss', avg_val_loss, step=global_step)
            tf.summary.scalar('conf_loss', avg_val_conf_loss, step=global_step)
            tf.summary.scalar('loc_loss', avg_val_loc_loss, step=global_step)
            gboxes, gclasses, gscores = post_process(gt_confs.numpy()[0], gt_locs.numpy()[0], gt_scores.numpy()[0], default_boxes)
            boxes, classes, scores = post_process(val_confs.numpy()[0], val_locs.numpy()[0],None, default_boxes, mode=2)
            img_gt = cv2.cvtColor(np.uint8(o_imgs[0]), cv2.COLOR_RGB2BGR)
            img_pred = cv2.cvtColor(np.uint8(o_imgs[0]), cv2.COLOR_RGB2BGR)
            draw_image(img_gt, gboxes, gclasses)
            draw_image(img_pred, boxes ,classes, mode='pred')
            tf.summary.image("val/gt", np.expand_dims(img_gt, 0),step=global_step)
            tf.summary.image("val/pred", np.expand_dims(img_pred, 0),step=global_step)
        
        ssd.save_weights(
            os.path.join(args.checkpoint_dir, 'ssd_epoch_{}.h5'.format(epoch + 1)))


