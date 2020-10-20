from tensorflow.keras import Model
from tensorflow.keras.applications import VGG16
import tensorflow.keras.layers as layers
import tensorflow as tf
import numpy as np
import os
import tensorflow.keras as keras
from layers import create_vgg16_layers, create_extra_layers, create_conf_head_layers, create_loc_head_layers, get_ssd_model, BatchNormalization


class SSD():
    """ Class for SSD model
    Attributes:
        num_classes: number of classes
    """

    def __init__(self, num_classes, arch='SSD512', batch_size=12):
        super(SSD, self).__init__()
        self.num_classes = num_classes
        self.model= get_ssd_model
        self.arch = arch
        self.batch_norm = BatchNormalization()
        self.batch_size= batch_size
        # self.extra_layers = create_extra_layers()
        self.conf_head_layers = create_conf_head_layers(num_classes)
        self.loc_head_layers = create_loc_head_layers()

        if arch == 'SSD300':
            # self.extra_layers.pop(-1)
            self.conf_head_layers.pop(-2)
            self.loc_head_layers.pop(-2)
            self.input_shape = [300,300,3]
        elif(arch == "SSD512"):
            self.input_shape = [512,512,3]
            
            
    def compute_heads(self, x, idx):
        """ Compute outputs of classification and regression heads
        Args:
            x: the input feature map
            idx: index of the head layer
        Returns:
            conf: output of the idx-th classification head
            loc: output of the idx-th regression head
        """
        conf = self.conf_head_layers[idx](x)
        
        # shape = tf.shape(conf)
        # print('reshaping', conf.shape)
        # conf = tf.reshape(conf, [conf.get_shape().as_list()[0] , -1, self.num_classes])

        loc = self.loc_head_layers[idx](x)
        # print('reshaping', loc.get_shape().as_list())
        # loc = tf.reshape(loc, [loc.get_shape().as_list()[0], -1, 4])

        return conf, loc


        
    def build_model(self, ):
        """ The forward pass
        Args:
            x: the input image
        Returns:
            confs: list of outputs of all classification heads
            locs: list of outputs of all regression heads
        """
        x_input = keras.Input(shape=self.input_shape,batch_size=self.batch_size)
        confs = []
        locs = []
        x = x_input
        x, out_layers = self.model(x, self.arch)
        for i in range(len(out_layers)):
            if( i == 0):                
                conf, loc = self.compute_heads(out_layers[i], i)
            else:
                conf, loc = self.compute_heads(out_layers[i], i)
            print(i, "\t", out_layers[i].shape)
            confs.append(conf)
            locs.append(loc)
        # confs = tf.concat(confs, axis=1)
        # locs = tf.concat(locs, axis=1)
        model = keras.Model(inputs=x_input, outputs=[*confs,*locs])
        return model
    
    def build_model_quant(self, ):
        """ The forward pass
        Args:
            x: the input image
        Returns:
            confs: list of outputs of all classification heads
            locs: list of outputs of all regression heads
        """
        x_input = keras.Input(shape=self.input_shape,batch_size=self.batch_size)
        confs = []
        locs = []
        x = x_input
        x, out_layers = self.model(x, self.arch)
        for i in range(len(out_layers)):
            if( i == 0):                
                conf, loc = self.compute_heads(out_layers[i], i)
            else:
                conf, loc = self.compute_heads(out_layers[i], i)
            conf = tf.reshape(conf, [conf.get_shape().as_list()[0] , -1, self.num_classes])
            loc = tf.reshape(loc, [loc.get_shape().as_list()[0], -1, 4])
            confs.append(conf)
            locs.append(loc)
        confs = tf.concat(confs, axis=1)
        locs = tf.concat(locs, axis=1)
        model = keras.Model(inputs=x_input, outputs=[confs,locs])
        return model

def create_ssd(num_classes, arch,batch_size, pretrained_type,
               checkpoint_dir=None,
               checkpoint_path=None ,scope=None, freeze=1, quant=1):
    """ Create SSD model and load pretrained weights
    Args:
        num_classes: number of classes
        pretrained_type: type of pretrained weights, can be either 'VGG16' or 'ssd'
        weight_path: path to pretrained weights
    Returns:
        net: the SSD model
    """
    
    # if(scope is None):
    net = SSD(num_classes, arch, batch_size=batch_size)
    if(quant > 0):
        model = net.build_model_quant()    
    else:
        model = net.build_model()
    # else:        
        # with scope.scope():
            # net = SSD(num_classes, arch)
            # model = net.build_model()
    # if(arch == "ssd300"):
        # net(tf.random.normal((1, 300, 300, 3)))
    if pretrained_type == 'base':
        origin_vgg = VGG16(weights='imagenet')        
        for i in range(len(origin_vgg.layers) - 5):
            # print(i , model.get_layer(index=i).name, origin_vgg.get_layer(index=i).name)        
            if(i < 14):
                model.get_layer(index=i).set_weights(
                    origin_vgg.get_layer(index=i).get_weights())
                if(freeze > 0):
                    model.get_layer(index=i).trainable = False
            elif(i > 15):
                model.get_layer(index=i+2).set_weights(
                    origin_vgg.get_layer(index=i).get_weights())
                if(freeze > 0):
                    model.get_layer(index=i).trainable = False
    elif pretrained_type == 'latest':
        try:
            paths = [os.path.join(checkpoint_dir, path)
                      for path in os.listdir(checkpoint_dir)]
            latest = sorted(paths, key=os.path.getmtime)[-1]
            model.load_weights(latest)
            print('model loaded from', latest)
        except AttributeError as e:
            print('Please make sure there is at least one checkpoint at {}'.format(
                checkpoint_dir))
            print('The model will be loaded from base weights.')            
    elif pretrained_type == 'specified':
        if not os.path.isfile(checkpoint_path):
            raise ValueError(
                'Not a valid checkpoint file: {}'.format(checkpoint_path))

        try:
            model.load_weights(checkpoint_path)
        except Exception as e:
            raise ValueError(
                'Please check the following\n1./ Is the path correct: {}?\n2./ Is the model architecture correct: {}?'.format(
                    checkpoint_path, arch))
    else:
        pass#raise ValueError('Unknown pretrained type: {}'.format(pretrained_type))

    return model

