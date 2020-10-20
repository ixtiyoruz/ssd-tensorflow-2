# -*- coding: utf-8 -*-
import os
os.environ['CUDA_VISIBLE_DEVICES'] = '3'
from tensorflow.keras.applications import VGG16
import tensorflow_model_optimization as tfmot
import tensorflow as tf
import numpy as np
from tensorflow import keras
from tensorflow.keras import layers
quantize_model = tfmot.quantization.keras.quantize_model
from anchor import generate_default_boxes, get_anchors, get_default_params
from box_utils import decode, compute_nms
from network_new import create_ssd
import yaml
import cv2

NUM_CLASSES = 2

# class QuantazitionHelper:
    # def __init__(self,):
        
with open('./config.yml') as f:
    cfg = yaml.load(f)

try:
    config = cfg['SSD300']#[args.arch.upper()]
except AttributeError:
    raise ValueError('Unknown architecture:')        
default_boxes = generate_default_boxes(config)
np.savetxt("default_boxes.csv", default_boxes.numpy(), delimiter=",")

model_orig = create_ssd(NUM_CLASSES, 'ssd300',1,
                        'latest',
                        './checkpoints', quant=True)


# img = cv2.imread("/home/essys/Pictures/picture2.jpeg", -1)
# img = cv2.resize(img, (300,300))
# img = img/127 -1.0
# res = model_orig(np.expand_dims(img, 0))

    #     self.create_output()
    # def create_output()

# outputs = model_orig.output
# newres = decode(default_boxes, outputs[1])
# confs = outputs[0]
# confs = tf.math.softmax(confs, axis=-1)
# # classes = tf.math.argmax(confs, axis=-1)
# # scores = tf.math.reduce_max(confs, axis=-1)
# model = keras.Model(inputs=model_orig.input, outputs=[confs, newres])
# model.load_weights("/home/essys/Documents/projects/ssd-tf2/checkpoints/ssd_epoch_1_step_400.h5")
# q_aware_model = quantize_model(model)



images_dir = '/media/essys/bd21e577-e5db-47a3-8845-f27be3f083a4/dataset/coco/images/val2017'
IMAGE_SIZE = 300
def representative_data_gen():

    dataset_list = tf.data.Dataset.list_files(images_dir + '/*')
    for i in range(2000):
        print(i)
        image = next(iter(dataset_list))
        image = tf.io.read_file(image)
        image = tf.io.decode_jpeg(image, channels=3)
        image = tf.image.resize(image, [IMAGE_SIZE, IMAGE_SIZE])        
        image = tf.cast(image/127.0 - 1.0, tf.float32)
        image = tf.expand_dims(image, 0)
        yield [image]

# saved_keras_model = 'model.h5'
# model.save(saved_keras_model)

converter = tf.lite.TFLiteConverter.from_keras_model(model_orig)
converter.optimizations = [tf.lite.Optimize.DEFAULT]
# This ensures that if any ops can't be quantized, the converter throws an error
converter.target_spec.supported_ops = [tf.lite.OpsSet.TFLITE_BUILTINS_INT8]
# These set the input and output tensors to uint8
# converter.allow_custom_ops = True
# converter.experimental_new_converter = False
converter.inference_input_type = tf.uint8
converter.inference_output_type = tf.uint8
# converter.inference_type = tf.uint8
# And this sets the representative dataset so we can quantize the activations
converter.representative_dataset = representative_data_gen
tflite_model = converter.convert()

# with open('ssd.quant.tflite', 'wb') as f:
  # f.write(tflite_model) 
import pathlib
tflite_model_file = pathlib.Path("./ssd.quant.tflite")
tflite_model_file.write_bytes(tflite_model)


optimized_model = tf.lite.Interpreter("ssd.quant.tflite")


print(optimized_model.get_input_details())
print(optimized_model.get_output_details())
optimized_model.allocate_tensors()
types = [optimized_model.get_tensor(i).dtype for i in range(160)]



# origin_vgg = VGG16(weights='imagenet')

# model = keras.Model(origin_vgg.get_layer(index=0).input, origin_vgg.get_layer(index=1).output)
# model  = keras.Model(model.input, model.output)


# origin_vgg = VGG16(weights='imagenet')
# for i in range(len(model.layers)):
#     model.get_layer(index=i).set_weights(
#         origin_vgg.get_layer(index=i).get_weights())


# inputs = keras.Input(shape=(784,))
# # img_inputs = keras.Input(shape=(32, 32, 3))

# dense = layers.Dense(64, activation="relu")
# x = dense(inputs)
# x = layers.Dense(64, activation="relu")(x)
# outputs = layers.Dense(10)(x)
# outputs = tf.reshape(outputs, [-1, 2, 5])
# model = keras.Model(inputs=inputs, outputs=outputs, name="mnist_model")

# # keras.utils.plot_model(model, "my_first_model.png")


# q_aware_model = quantize_model(model)

# y_true = tf.constant([[1, 2, 0, 2], [0, 2, 1, 0]])
# y_pred = tf.constant([[[0.05, 0.95, 0], [0.1, 0.8, 0.1], [0.9, 0.8, 0.1], [0.1, 0.8, 0.1]], 
#                       [[0.01, 0.95, 0], [0.8, 0.5, 0.1], [0.9, 0.2, 0.1], [0.5, 0.8, 0.1]]])
# # Using 'auto'/'sum_over_batch_size' reduction type.
# scce = tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True, reduction='none')
# loss = scce(y_true, y_pred).numpy()


# rank = tf.argsort(loss, axis=1, direction='DESCENDING')
# rank = tf.argsort(rank, axis=1)
# # neg_idx = rank < tf.expand_dims(num_neg, 1)
