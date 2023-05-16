import numpy as np
import tensorflow as tf
from tensorflow.python.keras import layers
from tensorflow.python.keras import initializers
import matplotlib.pyplot as plt
from tensorflow.keras.models import Model
from tensorflow.keras.layers import Input
from tensorflow.keras.applications import VGG16
import fGRU

# Load the VGG16 model with pre-trained weights
vgg_model = VGG16(weights='imagenet', include_top=False)
for layer in vgg_model.layers:
    if 'conv' in layer.name:
        layer.trainable = False

class GammaNetBSDS(tf.keras.Model):
    '''
    Gamma-net class
    '''
    def __init__(self, batch_size=1, steps=8, mode='segmentation'):
        super().__init__()
        self.batch_size = batch_size
        self.n_layers = 5
        self.steps = steps
        self.blocks = [] # stores gammanetblocks, number of items equals number 
                         # of layers, each layer contains one or two blocks.
        self.mode = mode

        # bottom-up layers
        self.blocks.append([[vgg_model.layers[1], vgg_model.layers[2], vgg_model.layers[3]]])
        self.blocks.append([[vgg_model.layers[4], vgg_model.layers[5], 
                             layers.Conv2D(
            filters=24, kernel_size = 3, strides = 1, padding = 'same', activation=tf.keras.activations.elu),
                             fGRU.fGRU(input_shape = [self.batch_size, 160, 240, 24], name='bu'), 
                             layers.Conv2D(
            filters=128, kernel_size = 3, strides = 1, padding = 'same', activation=tf.keras.activations.elu),
                             vgg_model.layers[6]]])
        self.blocks.append([[vgg_model.layers[7], vgg_model.layers[8], vgg_model.layers[9], 
                             layers.Conv2D(
            filters=36, kernel_size = 3, strides = 1, padding = 'same', activation=tf.keras.activations.elu),
                             fGRU.fGRU(input_shape = [self.batch_size, 80, 120, 36], name='bu'), 
                             layers.Conv2D(
            filters=256, kernel_size = 3, strides = 1, padding = 'same', activation=tf.keras.activations.elu),
                             vgg_model.layers[10]]])
        self.blocks.append([[vgg_model.layers[11], vgg_model.layers[12], vgg_model.layers[13], 
                             layers.Conv2D(
            filters=48, kernel_size = 3, strides = 1, padding = 'same', activation=tf.keras.activations.elu),
                             fGRU.fGRU(input_shape = [self.batch_size, 40, 60, 48], name='bu'), 
                             layers.Conv2D(
            filters=512, kernel_size = 3, strides = 1, padding = 'same', activation=tf.keras.activations.elu),
                             vgg_model.layers[14]]])
        self.blocks.append([[vgg_model.layers[15], vgg_model.layers[16], vgg_model.layers[17], 
                             layers.Conv2D(
            filters=64, kernel_size = 3, strides = 1, padding = 'same', activation=tf.keras.activations.elu),
                             fGRU.fGRU(input_shape = [self.batch_size, 20, 30, 64], name='bu')]])
        
        # top_down layers
        self.blocks[3].append([fGRU.InstanceNorm(64), 
                               tf.keras.layers.Resizing(40, 60), 
                               layers.Conv2D(filters=8, kernel_size=1, strides=1, activation='ReLU'),
                               layers.Conv2D(filters=48, kernel_size=1, strides=1, activation='ReLU'),
                               fGRU.fGRU(input_shape=[self.batch_size, 40, 60, 48], kernel_size=1, name='td')])
        self.blocks[2].append([fGRU.InstanceNorm(48), 
                               tf.keras.layers.Resizing(80, 120), 
                               layers.Conv2D(filters=8, kernel_size=1, strides=1, activation='ReLU'),
                               layers.Conv2D(filters=36, kernel_size=1, strides=1, activation='ReLU'),
                               fGRU.fGRU(input_shape=[self.batch_size, 80, 120, 36], kernel_size=1, name='td')])
        self.blocks[1].append([fGRU.InstanceNorm(36), 
                               tf.keras.layers.Resizing(160, 240), 
                               layers.Conv2D(filters=16, kernel_size=1, strides=1, activation='ReLU'),
                               layers.Conv2D(filters=24, kernel_size=1, strides=1, activation='ReLU'),
                               fGRU.fGRUfGRU(input_shape=[self.batch_size, 160, 240, 24], kernel_size=1, name='td')])
        # readout layer
        if self.mode == 'segmentation':
          self.blocks.append([[fGRU.InstanceNorm(24), 
                               tf.keras.layers.Resizing(320, 480), 
                               layers.Conv2D(filters=1, kernel_size=1, strides=1)]])
        else: pass

    def call(self, x):
        # pupulate a list that holds all the hidden states
        hidden_states = [tf.zeros([self.batch_size, 160, 240, 24]), 
                         tf.zeros([self.batch_size, 80, 120, 36]),
                         tf.zeros([self.batch_size, 40, 60, 48]),
                         tf.zeros([self.batch_size, 20, 30, 64])]

        # run through the alrgoithm
        for i in range(self.steps):
            z = x
            for layer in self.blocks[0][0]:
                z = layer(z)
            # In the paper, this assignment appears before the time loop,
            # and updates z on the first layer with ReLU and Conv every time 
            # step. 
            # This doesn't make much sense, because at time t, the input
            # to the first layer would already gone through t-1 ReLU and Convs,
            # but when you consider human brain, every second comes a fresh image
            # from the very bottom of the visual path.
            for l in range(1, self.n_layers):
                # bottom-up
                for layer in self.blocks[l][0]:
                    if layer.name == 'bu':
                        z = layer(z, hidden_states[l-1])
                        hidden_states[l-1] = z
                    else: z = layer(z)
            if i == self.steps - 1:
                z_class = z
            
            for l in range(self.n_layers-2, 0, -1):
            # top-down
                for layer in self.blocks[l][1]:
                    if layer.name == 'td':
                        z = layer(z, hidden_states[l-1])
                        hidden_states[l-1] = z
                    else: z = layer(z)
        if self.mode == 'classification':
            out = z_class
        else: out = hidden_states[0]
        for layer in self.blocks[-1][0]:
            out = layer(out)
        return out

################################################################################

# loss fuctions

def pixel_wise_bce(y_true, y_pred):
    # Flatten the prediction tensor and the true tensor
    y_pred_flat = tf.reshape(y_pred, [-1])
    y_true_flat = tf.reshape(y_true, [-1])

    # Compute binary cross entropy
    return tf.keras.losses.binary_crossentropy(y_true_flat, y_pred_flat)


# this loss function is not used eventually.
from tensorflow.keras import backend as K
# Helper function to enable loss function to be flexibly used for 
# both 2D or 3D image segmentation - source: https://github.com/frankkramer-lab/MIScnn
def identify_axis(shape):
    # Three dimensional
    if len(shape) == 5 : return [1,2,3]
    # Two dimensional
    elif len(shape) == 4 : return [1,2]
    # Exception - Unknown
    else : raise ValueError('Metric: Shape of tensor is neither 2D or 3D.')
    
def focal_tversky_loss(delta=0.7, gamma=0.75, smooth=0.000001):
    """A Novel Focal Tversky loss function with improved Attention U-Net for lesion segmentation
    Link: https://arxiv.org/abs/1810.07842
    Parameters
    ----------
    gamma : float, optional
        focal parameter controls degree of down-weighting of easy examples, by default 0.75
    """
    def loss_function(y_true, y_pred):
        # Clip values to prevent division by zero error
        epsilon = K.epsilon()
        y_pred = K.clip(y_pred, epsilon, 1. - epsilon) 
        axis = identify_axis(y_true.get_shape())
        # Calculate true positives (tp), false negatives (fn) and false positives (fp)     
        tp = K.sum(y_true * y_pred, axis=axis)
        fn = K.sum(y_true * (1-y_pred), axis=axis)
        fp = K.sum((1-y_true) * y_pred, axis=axis)
        tversky_class = (tp + smooth)/(tp + delta*fn + (1-delta)*fp + smooth)
        # Average class scores
        focal_tversky_loss = K.mean(K.pow((1-tversky_class), gamma))
	
        return focal_tversky_loss

    return loss_function