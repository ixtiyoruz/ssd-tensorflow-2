# -*- coding: utf-8 -*-
# %matplotlib inline
import matplotlib
import matplotlib.pyplot as plt
import tensorflow as tf
import tensorflow_datasets as tfds
import cv2
from box_utils import compute_target
from image_utils import random_patching, horizontal_flip
import numpy as np
from functools import partial
class COCOdataset:
    def __init__(self, default_boxes, year='2017', batch_size =32, input_shape = [512,512,3],  augmentation=None):
        self.default_boxes = default_boxes
        self.year = year   
        self.prepare_dataset()
        self.batch_size = batch_size
        self.input_shape = input_shape
        if augmentation == None:
            self.augmentation = ['original']
        else:
            self.augmentation = augmentation + ['original']
        # self.show()
    def prepare_dataset(self):

        
        self.builder = tfds.builder('coco/' +self.year)
        # 1. Create the tfrecord files (no-op if already exists)
        self.builder.download_and_prepare()
        # 2. Load the `tf.data.Dataset`
        self.train_dataset = self.builder.as_dataset(split='train', shuffle_files=True)
        self.val_dataset = self.builder.as_dataset(split='validation', shuffle_files=False)
        self.info = self.builder.info
        self.train_dataset = self.train_dataset
        # self.iterable_dataset = tf.compat.v1.data.make_initializable_iterator(self.train_dataset)
    
    def __len__(self):
        return self.info.splits['train'].num_examples
    
    def on_end_epoch(self):
        self.train_dataset = self.train_dataset.repeat().shuffle(2048)    

    def generate(self, mode="train"):
        """ The __getitem__ method
            so that the object can be iterable

        Args:

        Returns:
            img: tensor of shape (300, 300, 3)
            boxes: tensor of shape (num_gt, 4)
            labels: tensor of shape (num_gt,)
        """
        while(True):        
            self.on_end_epoch()
            for index in range(self.__len__()):
                imgs, gt_confs_all, gt_locs_all = [], [],[]
                for i in range(self.batch_size):
                    if(mode =='train'):
                        sample = next(tfds.as_numpy(self.train_dataset.take(self.batch_size)))
                    else:
                        sample = next(tfds.as_numpy(self.val_dataset.take(self.batch_size)))
                    img = sample['image']
                    boxes = sample['objects']['bbox']
                    labels = sample['objects']['label']
                    # orig_shape = img.shape
                    # # img, orig_shape = self._get_image(index)
                    # filename = indices[index]
                    # img = self._get_image(index)
                    # w, h = img.size
                    # boxes, labels = self._get_annotation(index, (h, w))
                    boxes = tf.constant(boxes, dtype=tf.float32)
                    labels = tf.constant(labels+1, dtype=tf.int64) # 0 is background
        
                    # augmentation_method = np.random.choice(self.augmentation)
                    # if augmentation_method == 'patch':
                    #     img, boxes, labels = random_patching(img, boxes, labels)
                    # elif augmentation_method == 'flip':
                    #     img, boxes, labels = horizontal_flip(img, boxes, labels)
                    # print((self.input_shape, self.input_shape))
                    img = cv2.resize(img, (self.input_shape, self.input_shape))
                    # img = np.array(img.resize(
                        # (self.input_shape, self.input_shape)), dtype=np.float32)
                    img = (img / 127.0) - 1.0
                    img = tf.constant(img, dtype=tf.float32)
                    # if(len(boxes) == 0):
                    gt_confs, gt_locs = compute_target(
                        self.default_boxes, boxes, labels)
                    imgs.append(img)
                    gt_confs_all.append(gt_confs)
                    gt_locs_all.append(gt_locs)
                    
                yield tf.stack(imgs), tf.stack(gt_confs_all), tf.stack(gt_locs_all)
    def show(self,):
        fig = tfds.show_examples(self.ds, self.info)
        
    def show_images_bbox(self):
        samples = self.val_dataset.take(10)
        for sample in samples:
            img = sample['image'].numpy()
            boxes = np.int32(sample['objects']['bbox']*np.array([*img.shape[:-1], *img.shape[:-1]]))
            labels = sample['objects']['label']        
            # w, h = img.size
            print(labels+1) # 0 is background
            for i in range(len(boxes)):
                cv2.rectangle(img, tuple(boxes[i][:2][::-1]), tuple(boxes[i][2:][::-1]), (255,0,0))
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
    
    
    
if __name__ == '__main__':
    datasetbuilder = COCOdataset(None)
    datasetbuilder.prepare_dataset()
    datasetbuilder.show_images_bbox()