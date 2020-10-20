import tensorflow as tf
import os
import numpy as np
import xml.etree.ElementTree as ET
from PIL import Image
import random
from anchor import generate_default_boxes
from box_utils import compute_target
from image_utils import random_patching, horizontal_flip
from functools import partial
import cv2

class VOCDataset():
    """ Class for VOC Dataset

    Attributes:
        root_dir: dataset root dir (ex: ./data/VOCdevkit)
        year: dataset's year (2007 or 2012)
        num_examples: number of examples to be used
                      (in case one wants to overfit small data)
    """

    def __init__(self, root_dir, year, default_boxes,
                 new_size, num_examples=-1,batch_size=12, augmentation=None):
        super(VOCDataset, self).__init__()
        self.idx_to_name = [
            'aeroplane', 'bicycle', 'bird', 'boat',
            'bottle', 'bus', 'car', 'cat', 'chair',
            'cow', 'diningtable', 'dog', 'horse',
            'motorbike', 'person', 'pottedplant',
            'sheep', 'sofa', 'train', 'tvmonitor']
        self.name_to_idx = dict([(v, k)
                                 for k, v in enumerate(self.idx_to_name)])

        self.data_dir = os.path.join(root_dir, 'VOC{}'.format(year))
        self.image_dir = os.path.join(self.data_dir, 'JPEGImages')
        self.anno_dir = os.path.join(self.data_dir, 'Annotations')
        self.ids = list(map(lambda x: x[:-4], os.listdir(self.image_dir)))
        self.default_boxes = default_boxes
        self.new_size = new_size
        self.batch_size = batch_size
        if num_examples != -1:
            self.ids = self.ids[:num_examples]

        self.train_ids = self.ids[:int(len(self.ids) * 0.75)]
        self.val_ids = self.ids[int(len(self.ids) * 0.75):]

        if augmentation == None:
            self.augmentation = ['original']
        else:
            self.augmentation = augmentation + ['original']

    def __len__(self):
        return len(self.ids)

    def _get_image(self, index):
        """ Method to read image from file
            then resize to (300, 300)
            then subtract by ImageNet's mean
            then convert to Tensor

        Args:
            index: the index to get filename from self.ids

        Returns:
            img: tensor of shape (3, 300, 300)
        """
        filename = self.ids[index]
        img_path = os.path.join(self.image_dir, filename + '.jpg')
        img = Image.open(img_path)

        return img

    def _get_annotation(self, index, orig_shape):
        """ Method to read annotation from file
            Boxes are normalized to image size
            Integer labels are increased by 1

        Args:
            index: the index to get filename from self.ids
            orig_shape: image's original shape

        Returns:
            boxes: numpy array of shape (num_gt, 4)
            labels: numpy array of shape (num_gt,)
        """
        h, w = orig_shape
        filename = self.ids[index]
        anno_path = os.path.join(self.anno_dir, filename + '.xml')
        objects = ET.parse(anno_path).findall('object')
        boxes = []
        labels = []

        for obj in objects:
            name = obj.find('name').text.lower().strip()
            if(name in ['person']):
                bndbox = obj.find('bndbox')
                xmin = (float(bndbox.find('xmin').text) - 1) / w
                ymin = (float(bndbox.find('ymin').text) - 1) / h
                xmax = (float(bndbox.find('xmax').text) - 1) / w
                ymax = (float(bndbox.find('ymax').text) - 1) / h
                boxes.append([xmin, ymin, xmax, ymax])            
                labels.append(1)                            

        return np.array(boxes, dtype=np.float32), np.array(labels, dtype=np.int64)

    def generate(self, subset=None, num=20000):
        """ The __getitem__ method
            so that the object can be iterable

        Args:
            index: the index to get filename from self.ids

        Returns:
            img: tensor of shape (300, 300, 3)
            boxes: tensor of shape (num_gt, 4)
            labels: tensor of shape (num_gt,)
        """
        if subset == 'train':
            indices = self.train_ids
        elif subset == 'val':
            indices = self.val_ids
        else:
            indices = self.ids
        indices = np.array(indices)
        counter = 0
        while(counter < num):  
            counter = counter  +1
            itemp = np.random.randint(0,len(indices), size=[self.batch_size])
            # indexes = indices[itemp]
            # img, orig_shape = self._get_image(index)
            # filename = indices[index]
            imgs, gt_confs_all, gt_scores_all, gt_locs_all, o_imgs, o_boxes, o_labels = [], [],[], [], [], [], []
            for index in itemp:
                img = self._get_image(index)
                w, h = img.size
                boxes, labels = self._get_annotation(index, (h, w))
                boxes = tf.constant(boxes, dtype=tf.float32)
                labels = tf.constant(labels, dtype=tf.int64)
                
                augmentation_method = np.random.choice(self.augmentation)
                
                if augmentation_method == 'patch':
                    img, boxes, labels = random_patching(img, boxes, labels)
                    img  =np.float32(img)
                elif augmentation_method == 'flip':
                    img, boxes, labels = horizontal_flip(np.float32(img), boxes, labels)
                else:
                    img = np.array(img, dtype=np.float32)
                    
                # print(img.shape, img.dtype)
                img_o = cv2.resize(img,(self.new_size, self.new_size))
                
                img = (img_o / 127.0) - 1.0
                img = tf.constant(img, dtype=tf.float32)
    
                gt_confs, gt_locs, gt_scores= compute_target(
                    self.default_boxes, boxes, labels)
                imgs.append(img)
                gt_confs_all.append(gt_confs)
                gt_locs_all.append(gt_locs)
                gt_scores_all.append(gt_scores)
                o_imgs.append(img_o)
                o_boxes.append(boxes)
                o_labels.append(labels)
                # print('1', img.dtype, gt_confs.dtype, gt_locs.dtype)
            imgs, gt_confs_all, gt_locs_all =  tf.stack(imgs), tf.stack(gt_confs_all), tf.stack(gt_locs_all)
            # print('----', np.max(imgs.numpy()), np.max(gt_confs_all.numpy()), np.max(gt_locs_all)) 
            yield tf.stack(imgs), tf.stack(gt_confs_all), tf.stack(gt_locs_all),tf.stack(gt_scores_all), o_imgs, o_boxes, o_labels


# def create_batch_generator(root_dir, year, default_boxes,
#                            new_size, batch_size, num_batches,
#                            mode,
#                            augmentation=None):
#     num_examples = batch_size * num_batches if num_batches > 0 else -1
#     voc = VOCDataset(root_dir, year, default_boxes,
#                      new_size, num_examples, augmentation)

#     info = {
#         'idx_to_name': voc.idx_to_name,
#         'name_to_idx': voc.name_to_idx,
#         'length': len(voc),
#         'image_dir': voc.image_dir,
#         'anno_dir': voc.anno_dir
#     }

#     if mode == 'train':
#         train_gen = partial(voc.generate, subset='train')
#         train_dataset = tf.data.Dataset.from_generator(
#             train_gen, (tf.float32, tf.int64, tf.float32, tf.float32, tf.float32, tf.int64))
#         val_gen = partial(voc.generate, subset='val')
#         val_dataset = tf.data.Dataset.from_generator(
#             val_gen, (tf.float32, tf.int64, tf.float32, tf.float32, tf.float32, tf.int64 ))

#         train_dataset = train_dataset.shuffle(40).batch(batch_size)
#         val_dataset = val_dataset.batch(batch_size)

#         return train_dataset.take(num_batches), val_dataset.take(-1), info
#     else:
#         dataset = tf.data.Dataset.from_generator(
#             voc.generate, (tf.string, tf.float32, tf.int64, tf.float32))
#         dataset = dataset.batch(batch_size)
#         return dataset.take(num_batches), info
# import yaml
# with open('./config.yml') as f:
#     cfg = yaml.load(f)

# try:
#     config = cfg['SSD300']#[args.arch.upper()]
# except AttributeError:
#     pass
# default_boxes = generate_default_boxes(config)
# voc = VOCDataset('/media/essys/bd21e577-e5db-47a3-8845-f27be3f083a4/dataset/voc/VOCdevkit/', '2012', default_boxes, new_size=300)
