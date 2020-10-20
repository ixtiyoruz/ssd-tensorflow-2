# -*- coding: utf-8 -*-
# %matplotlib inline
import matplotlib
import matplotlib.pyplot as plt
import tensorflow as tf
import cv2
from box_utils import compute_target, tf_ssd_bboxes_encode, decode
from image_utils import random_patching, horizontal_flip
import numpy as np
from pycocotools.coco import COCO
from functools import partial
from random import shuffle
from box_utils import decode, compute_nms

class COCOdataset:
    def __init__(self,dataDir, default_boxes, datatype='val2017', batch_size =32, input_shape = [512,512,3],  augmentation=None):
        self.default_boxes = default_boxes
        self.datatype = datatype   
        
        self.batch_size = batch_size
        self.input_shape = input_shape
        self.dataDir = dataDir
        if augmentation == None:
            self.augmentation = ['original']
        else:
            self.augmentation = augmentation + ['original']
        self.cat_names = ["person"]
        self.prepare_dataset()
        # self.show()
    def prepare_dataset(self):
        annFile='{}/annotations/instances_{}.json'.format(self.dataDir,self.datatype)
        self.coco=COCO(annFile)
        self.catIds = self.coco.getCatIds(catNms=self.cat_names)
        self.imgIds = self.coco.getImgIds()
        shuffle(self.imgIds)
        self.global_index = 0
    def __len__(self):
        return len(self.imgIds)
    
    def on_end_epoch(self):
        shuffle(self.imgIds)

    def generate(self, num=20000):
        """ The __getitem__ method
            so that the object can be iterable

        Args:

        Returns:
            img: tensor of shape (300, 300, 3)
            boxes: tensor of shape (num_gt, 4)
            labels: tensor of shape (num_gt,)
        """
        counter = 0
        while(counter < num):  
            counter = counter  +1
            imgs, gt_confs_all, gt_locs_all,gt_scores_all, o_imgs, o_boxes, o_labels = [], [],[], [], [], [], []
            for i in range(self.batch_size):                
                img, boxes, labels, shape = None,None, None,None
                while(True):
                    self.global_index = self.global_index + 1
                    if(self.global_index == self.__len__()):
                        self.on_end_epoch()
                        self.global_index = 0
                    img, boxes, labels,shape = self.read_img(self.global_index)
                    if(img is not None):
                        if(len(labels) > 0):
                            break
                    
                  
                # orig_shape = img.shape
                # # img, orig_shape = self._get_image(index)
                # filename = indices[index]
                # img = self._get_image(index)
                # w, h = img.size
                # boxes, labels = self._get_annotation(index, (h, w))
                boxes = tf.constant(boxes, dtype=tf.float32)
                labels = tf.constant(labels, dtype=tf.int64)
    
                augmentation_method = np.random.choice(self.augmentation)
                if augmentation_method == 'patch':
                    # print('patching')
                    img, boxes, labels = random_patching(img, boxes, labels)                    
                elif augmentation_method == 'flip':
                    # print('flipping')
                    img, boxes, labels = horizontal_flip(img, boxes, labels)
                # print((self.input_shape, self.input_shape))
                img = cv2.resize(img, (self.input_shape[0], self.input_shape[1]))
                # img = np.array(img.resize(
                    # (self.input_shape, self.input_shape)), dtype=np.float32)
                o_imgs.append(img)
                img = img / 127.0 -1
                img = tf.constant(img, dtype=tf.float32)
                # if(len(boxes) == 0):
                gt_confs, gt_locs,gt_scores = compute_target(
                    self.default_boxes, boxes, labels)
                # gt_confs, gt_locs, gt_scores = tf_ssd_bboxes_encode(labels, boxes, self.default_boxes, len(self.cat_names) + 1, None)    
                
                imgs.append(img)
                gt_confs_all.append(gt_confs)
                gt_locs_all.append(gt_locs)
                gt_scores_all.append(gt_scores)
                o_boxes.append(boxes)
                o_labels.append(labels)
                
            yield tf.stack(imgs), tf.stack(gt_confs_all), tf.stack(gt_locs_all),tf.stack(gt_scores_all), o_imgs, o_boxes, o_labels
            
    def read_img(self, index):
        imginfo = self.coco.loadImgs(self.imgIds[index])[0]
        # imgname = imginfo['file_name']
        imgpath  = '%s/images/%s/%s'%(self.dataDir,self.datatype,imginfo['file_name'])
        img = cv2.imread(imgpath, -1)
        if(not len(np.shape(img)) == 3):
            return None, None,None,None
        annIds = self.coco.getAnnIds(imgIds=imginfo['id'], catIds=self.catIds, iscrowd=None)
        anns = self.coco.loadAnns(annIds)
        shape = np.shape(img)        
        bboxes, labels = self.get_bboxes(anns, shape)
        return img, bboxes, labels, shape
    def get_bboxes(self, anns, shape):
        
        h,w,ch = shape
        bboxes = []
        labels = []
        for i in range(len(anns)):
            category = anns[i]['category_id']
            if(category in self.catIds):
                xmin,ymin, w_,h_ = anns[i]['bbox']
                xmin,ymin,xmax,ymax = xmin/w, ymin/h,(xmin + w_)/w, (ymin + h_)/h
                bboxes.append([xmin,ymin,xmax,ymax])                        
                label = np.where(category== np.array(self.catIds))[0][0]
                labels.append(label+1) # 0 is backgroud            
            else:
                print(category)
        return bboxes, labels
    def show_processed_bboxes(self):        
        index = np.random.randint(0,len(self.imgIds))
        img , bboxes, labels,shape = self.read_img(index)
        h,w,ch = shape
        for i in range(len(bboxes)):
            bbox = bboxes[i]
            x1,y1, x2,y2 = bbox        
            x1,y1,x2,y2 = int(x1 * w), int(y1 * h), int(x2 * w), int(y2 * h)
            cv2.rectangle(img, (x1, y1), (x2, y2), (255,0,0), 2)
            cv2.putText(img,str(labels[i]), (x1,y1+10),1, 2.0, (0,0,255))
        cv2_im_processed = cv2.cvtColor(np.array(img), cv2.COLOR_RGB2BGR)
        plt.imshow(cv2_im_processed)
        plt.show()
    def show(self):
        index = np.random.randint(0,len(self.imgIds))
        imginfo= self.coco.loadImgs(self.imgIds[index])[0]
        
        annIds = self.coco.getAnnIds(imgIds=imginfo['id'], catIds=self.catIds, iscrowd=None)
        anns = self.coco.loadAnns(annIds)
        img = self.read_img(index)
        
        for i in range(len(anns)):
            bbox = anns[i]['bbox']
            x1,y1, w,h = np.int32(bbox)        
            img  = cv2.rectangle(img, (x1, y1), (x1+w, y1+h), (255,0,0), 2)
        plt.imshow(img)
        plt.show()
        

            
            
def create_batch_generator(year, default_boxes,
                           new_size, batch_size, num_batches,
                           mode,
                           augmentation=None):
    voc = COCOdataset(default_boxes,
                     input_shape=new_size, augmentation=augmentation)
    info = {
        # 'idx_to_name': voc.idx_to_name,
        # 'name_to_idx': voc.name_to_idx,
        'length': len(voc),
        # 'image_dir': voc.image_dir,
        # 'anno_dir': voc.anno_dir
    }

    if mode == 'train':
        train_gen =partial(voc.generate, mode='train')
        train_dataset = tf.data.Dataset.from_generator(
            train_gen, (tf.float32, tf.int64, tf.float32))
        val_gen = partial(voc.generate, mode='validation')
        val_dataset = tf.data.Dataset.from_generator(
            val_gen, (tf.float32, tf.int64, tf.float32))

        train_dataset = train_dataset.batch(batch_size).shuffle(32)
        val_dataset = val_dataset.batch(batch_size)

        return train_dataset.take(num_batches), val_dataset.take(num_batches), info
    else:
        dataset = tf.data.Dataset.from_generator(
            voc.generate, (tf.float32, tf.int64, tf.float32))
        dataset = dataset.batch(batch_size)
        return dataset.take(num_batches), info      
    
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
def post_process(confs, locs,scores, default_boxes, mode=1):
    # confs = tf.squeeze(confs, 0)
    # locs = tf.squeeze(locs, 0)
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
if __name__ == '__main__':
    import os
    from anchor import generate_default_boxes, get_anchors, get_default_params
    import np_methods
    import yaml
    os.environ['CUDA_VISIBLE_DEVICES'] = "1"
    NUM_CLASSES = 8
    # config = get_default_params()
    # config._replace(num_classes=NUM_CLASSES)
    # default_boxes =  get_anchors(config)
    # dfb = convert_default_boxes(default_boxes)
    
    
    with open('./config.yml') as f:
        cfg = yaml.load(f)

    try:
        config = cfg['SSD300']#[args.arch.upper()]
    except AttributeError:
        pass
    default_boxes = generate_default_boxes(config)
    
    voc = COCOdataset('/media/essys/bd21e577-e5db-47a3-8845-f27be3f083a4/dataset/coco',default_boxes,batch_size=12,datatype='train2017',
                      input_shape=[config['image_size'], config['image_size']], augmentation=['flip', 'patch'])
    voc.prepare_dataset()
    voc.show_processed_bboxes()
    import time 
    generator = voc.generate()
    dfb = default_boxes.numpy()
    for i in range(10):
        start = time.time()    
        imgs, gt_confs, gt_locs,gt_scores, o_imgs, o_boxes, o_labels = next(generator)
        img_gt = cv2.cvtColor(np.uint8(o_imgs[0]), cv2.COLOR_RGB2BGR)        
        # newres = decode(default_boxes, gt_locs.numpy()[0]).numpy()
        # rclasses, rscores, rbboxes = np_methods.bboxes_sort(gt_confs.numpy()[0], gt_scores.numpy()[0], newres, top_k=500)
        # rclasses, rscores, rbboxes = np_methods.bboxes_nms(rclasses, rscores, rbboxes, nms_threshold=.45)        
        rbboxes, rclasses, scores = post_process(gt_confs.numpy()[0], gt_locs.numpy()[0],gt_scores, default_boxes)
        draw_image(img_gt, rbboxes, rclasses)
        plt.imshow(img_gt)
        plt.show()
        print(time.time() - start)

# from PIL import Image

# im = Image.open("/home/essys/Pictures/img.jpg")

# # The crop method from the Image module takes four coordinates as input.
# # The right can also be represented as (left+width)
# # and lower can be represented as (upper+height).
# (left, upper, right, lower) = (20, 20, 100, 100)

# # Here the image "im" is cropped and assigned to new variable im_crop
# im_crop = im.crop((0.1, 0.02, 0.4, 0.17))
# img = np.array(im_crop)
# plt.imshow()
# plt.show()
# # im_crop.show()
