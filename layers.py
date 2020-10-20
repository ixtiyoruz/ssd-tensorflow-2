import tensorflow as tf
import tensorflow.keras.layers as layers
from tensorflow.keras import Sequential


def get_ssd_model(x, arch='SSD512'):    
    out_layers = []
    
    # first block out = [150, 150, 3], [256,256,3]
    x = layers.Conv2D(64, 3, padding='same', activation='relu')(x)
    x = layers.Conv2D(64, 3, padding='same', activation='relu')(x)
    x = layers.MaxPool2D(2, 2, padding='same')(x)
    
    # second block out = [75, 75, 3], [128,128,3]
    x = layers.Conv2D(128, 3, padding='same', activation='relu')(x)
    x = layers.Conv2D(128, 3, padding='same', activation='relu')(x)
    x = layers.MaxPool2D(2, 2, padding='same')(x)
    
    # third block out = [38, 38, 3], [64,64,3]
    x = layers.Conv2D(256, 3, padding='same', activation='relu')(x)
    x = layers.Conv2D(256, 3, padding='same', activation='relu')(x)
    x = layers.Conv2D(256, 3, padding='same', activation='relu')(x)
    x = layers.MaxPool2D(2, 2, padding='same')(x)
    
    # forth block out = [38, 38, 3], [64,64,3]
    x = layers.Conv2D(512, 3, padding='same', activation='relu')(x)
    x = layers.Conv2D(512, 3, padding='same', activation='relu')(x)
    x = layers.Conv2D(512, 3, padding='same')(x) # use this layer as first output
    x = layers.BatchNormalization()(x)
    out_layers.append(x)
    
    x = layers.Activation('relu')(x)
    # pooling out = [19, 19, 3], [32, 32, 3]
    x = layers.MaxPool2D(2, 2, padding='same')(x)
    
    # fifth block out = [10, 10, 3], [16, 16, 3]
    x = layers.Conv2D(512, 3, padding='same', activation='relu')(x)
    x = layers.Conv2D(512, 3, padding='same', activation='relu')(x)
    x = layers.Conv2D(512, 3, padding='same', activation='relu')(x) # up to here you can use pretrained vgg 16 weights        
    x = layers.MaxPool2D(3, 1, padding='same')(x)
    
    # 6th block (changed)
    # atrous conv2d for 6th block out = [10, 10, 3], [16, 16, 3]
    x = layers.Conv2D(1024, 3, padding='same', dilation_rate=6, activation='relu')(x)
    x = layers.Conv2D(1024, 1, padding='same', activation='relu')(x)
    out_layers.append(x)
    
    # skip the 7th block
    
    # 8th block  = [5, 5, 3], [8, 8, 3]
    x = layers.Conv2D(256, 1, activation='relu')(x)
    x = layers.Conv2D(512, 3, strides=2, padding='same', activation='relu')(x)
    out_layers.append(x)
    
    # 9th block  = [3, 3, 3], [4, 4, 3]
    x = layers.Conv2D(128, 1, activation='relu')(x)
    x = layers.Conv2D(256, 3, strides=2, padding='same', activation='relu')(x)
    out_layers.append(x)
    
    # 10th block output shape: B, 256, 3, 3
    
    x = layers.Conv2D(128, 1, activation='relu')(x)
    x = layers.Conv2D(256, 3, activation='relu')(x)
    out_layers.append(x)
    
    # 11th block output shape: B, 256, 1, 1
    
    x = layers.Conv2D(128, 1, activation='relu')(x)
    x = layers.Conv2D(256, 3, activation='relu')(x)
    out_layers.append(x)
    if(arch=='SSD512'):
        # # 12th block output shape: B, 256, 1, 1    
        x = layers.Conv2D(128, 1, activation='relu')(x)
        x = layers.Conv2D(256, 4, activation='relu')(x)
        out_layers.append(x)    
        
    return x, out_layers
    
    
def create_vgg16_layers():
    vgg16_conv4 = [
        layers.Conv2D(64, 3, padding='same', activation='relu'),
        layers.Conv2D(64, 3, padding='same', activation='relu'),
        layers.MaxPool2D(2, 2, padding='same'),

        layers.Conv2D(128, 3, padding='same', activation='relu'),
        layers.Conv2D(128, 3, padding='same', activation='relu'),
        layers.MaxPool2D(2, 2, padding='same'),

        layers.Conv2D(256, 3, padding='same', activation='relu'),
        layers.Conv2D(256, 3, padding='same', activation='relu'),
        layers.Conv2D(256, 3, padding='same', activation='relu'),
        layers.MaxPool2D(2, 2, padding='same'),

        layers.Conv2D(512, 3, padding='same', activation='relu'),
        layers.Conv2D(512, 3, padding='same', activation='relu'),
        layers.Conv2D(512, 3, padding='same', activation='relu'),
        layers.MaxPool2D(2, 2, padding='same'),

        layers.Conv2D(512, 3, padding='same', activation='relu'),
        layers.Conv2D(512, 3, padding='same', activation='relu'),
        layers.Conv2D(512, 3, padding='same', activation='relu'),
    ]
    
    x = layers.Input(shape=[None, None, 3])
    out = x
    for layer in vgg16_conv4:
        out = layer(out)

    vgg16_conv4 = tf.keras.Model(x, out)

   

    vgg16_conv7 = [
        # Difference from original VGG16:
        # 5th maxpool layer has kernel size = 3 and stride = 1
        layers.MaxPool2D(3, 1, padding='same'),
        # atrous conv2d for 6th block
        layers.Conv2D(1024, 3, padding='same',
                      dilation_rate=6, activation='relu'),
        layers.Conv2D(1024, 1, padding='same', activation='relu'),
    ]
        

    x = layers.Input(shape=[None, None, 512])
    out = x
    for layer in vgg16_conv7:
        out = layer(out)

    vgg16_conv7 = tf.keras.Model(x, out)

    return vgg16_conv4, vgg16_conv7


def create_extra_layers():
    """ Create extra layers
        8th to 11th blocks
    """
    extra_layers = [
        # 8th block output shape: B, 512, 10, 10
        Sequential([
            layers.Conv2D(256, 1, activation='relu'),
            layers.Conv2D(512, 3, strides=2, padding='same',
                          activation='relu'),
        ]),
        # 9th block output shape: B, 256, 5, 5
        Sequential([
            layers.Conv2D(128, 1, activation='relu'),
            layers.Conv2D(256, 3, strides=2, padding='same',
                          activation='relu'),
        ]),
        # 10th block output shape: B, 256, 3, 3
        Sequential([
            layers.Conv2D(128, 1, activation='relu'),
            layers.Conv2D(256, 3, activation='relu'),
        ]),
        # 11th block output shape: B, 256, 1, 1
        Sequential([
            layers.Conv2D(128, 1, activation='relu'),
            layers.Conv2D(256, 3, activation='relu'),
        ]),
        # 12th block output shape: B, 256, 1, 1
        Sequential([
            layers.Conv2D(128, 1, activation='relu'),
            layers.Conv2D(256, 4, activation='relu'),
        ])
    ]

    return extra_layers


def create_conf_head_layers(num_classes):
    """ Create layers for classification
    """
    conf_head_layers = [
        layers.Conv2D(4 * num_classes, kernel_size=3,
                      padding='same'),  # for 4th block
        layers.Conv2D(6 * num_classes, kernel_size=3,
                      padding='same'),  # for 7th block
        layers.Conv2D(6 * num_classes, kernel_size=3,
                      padding='same'),  # for 8th block
        layers.Conv2D(6 * num_classes, kernel_size=3,
                      padding='same'),  # for 9th block
        layers.Conv2D(4 * num_classes, kernel_size=3,
                      padding='same'),  # for 10th block
        layers.Conv2D(4 * num_classes, kernel_size=3,
                      padding='same'),  # for 11th block
        layers.Conv2D(4 * num_classes, kernel_size=1)  # for 12th block
    ]

    return conf_head_layers


def create_loc_head_layers():
    """ Create layers for regression
    """
    loc_head_layers = [
        layers.Conv2D(4 * 4, kernel_size=3, padding='same'),
        layers.Conv2D(6 * 4, kernel_size=3, padding='same'),
        layers.Conv2D(6 * 4, kernel_size=3, padding='same'),
        layers.Conv2D(6 * 4, kernel_size=3, padding='same'),
        layers.Conv2D(4 * 4, kernel_size=3, padding='same'),
        layers.Conv2D(4 * 4, kernel_size=3, padding='same'),
        layers.Conv2D(4 * 4, kernel_size=1)
    ]

    return loc_head_layers

class BatchNormalization(tf.keras.layers.BatchNormalization):
    """
    "Frozen state" and "inference mode" are two separate concepts.
    `layer.trainable = False` is to freeze the layer, so the layer will use
    stored moving `var` and `mean` in the "inference mode", and both `gama`
    and `beta` will not be updated !
    """
    def call(self, x, training=False):
        if not training:
            training = tf.constant(False)
        training = tf.logical_and(training, self.trainable)
        return super().call(x, training)