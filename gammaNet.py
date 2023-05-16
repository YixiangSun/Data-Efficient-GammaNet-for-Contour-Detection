import tensorflow as tf
from tensorflow.python.keras import layers
import fGRU

class GammaNetBlock(layers.Layer):
    '''
    Generate a block in gamma-net
    '''
    def __init__(self, batch_size, input_shape, layers_config, name = 'block'):
        '''
        params: 
        input_channels: int, the number of input channels
        hidden_channels: int, the number of channels within the block
        layers: a list of tuples, specifying what kind of layers it contains:
                Conv2D: ('c', [kernel_size, strides])
                TransposedConv2D: ('t', [kernel_size, strides])
                fGru: ('f', [input_shape, use_attention])
                maxPool: ('m', [kernel_size, strides])
                instanceNorm: ('i')
                denselayer: ('d', [unit])
        '''
        super().__init__()
        self.batch_size = batch_size

        self.input_shape_ = [batch_size, input_shape[0], input_shape[1], input_shape[2]]
        # here, input_shape_ is in [batch_size, height, width, channel_size]

        self.hidden_channels = self.input_shape_[-1]
        self.fgru = None
        self.layers_config = layers_config
        self.block_prefix = name + '_'
        self.layers = []

        for layer in layers_config:
        # populate the blocks with layers
            if layer[0] == 'c':
                kernel_size = layer[1][0]
                strides = layer[1][1]
                name = self.block_prefix + 'conv'
                self.layers.append(layers.Conv2D(
                    filters=self.hidden_channels, 
                    kernel_size=kernel_size, 
                    strides=strides,
                    padding='same',
                    activation='ReLU', 
                    name=name
                    ))
                
            elif layer[0] == 't':
                kernel_size = layer[1][0]
                strides = layer[1][1]
                name = self.block_prefix + 'tconv'
                self.layers.append(layers.Conv2DTranspose(
                    filters=self.hidden_channels, 
                    kernel_size=kernel_size, 
                    strides=strides,
                    padding='same', 
                    name=name
                    ))
                
            elif layer[0] == 'f':
                kernel_size = layer[1][0]
                use_attention = layer[1][1]
                name = self.block_prefix + 'fGRU'
                self.fgru=fGRU.fGRU(
                    input_shape=self.input_shape_, 
                    kernel_size=kernel_size,
                    use_attention = use_attention,
                    name = name
                    )
                self.layers.append(self.fgru)

            elif layer[0] == 'm':
                pool_size = layer[1][0]
                strides = layer[1][1]
                name = self.block_prefix + 'maxpool'
                self.layers.append(layers.MaxPool2D(
                    pool_size=pool_size, 
                    strides=strides,
                    padding='valid', 
                    name = name
                    ))
                
            elif layer[0] == 'i':
                name = self.block_prefix + 'instance_norm'
                self.layers.append(fGRU.InstanceNorm(self.hidden_channels, name = name))

            elif layer[0] == 'd':
                unit = layer[1][0]
                name = self.block_prefix + 'dense'
                if layer[1][1] == 's':
                    self.layers.append(layers.Dense(unit, activation='softmax', name = name))
                elif layer[1][1] == 'l':
                    self.layers.append(layers.Dense(unit, activation='leaky_relu', name = name))

            elif layer[0] == 'l':
                name = self.block_prefix + 'flatten'
                self.layers.append(layers.Flatten(name = name))

    def call(self, input, hidden_state):
        # take in an input and the hidden state of the fGRU at this block
        z = input
        h = None # default to None when used as placeholder for readout block
        for layer in self.layers:
            # update the hidden state by passing it through the fGRU unit
            if layer == self.fgru:
                z = layer(z, hidden_state)
                h = z
            else:
                z = layer(z)
        # return the output and the updated hidden state of the fGRU at this block
        return z, h

# in each layer, there are three lists: 
# input shape (without batch_size), bottom-up unit, top-down unit (if any)
default_config = [
    [[384, 384, 24], 
     [('c', [3, 1]), ('c', [3, 1]), ('f', [9, False]), ('m', [2, 2])],
     [('t', [4, 2]), ('c', [3, 1]), ('i'), ('f', [1, False])]], # first layer
    [[192, 192, 28], 
     [('c', [3, 1]), ('f', [7, False]), ('m', [2, 2])],
     [('t', [4, 2]), ('c', [3, 1]), ('i'), ('f', [1, False])]], # second layer
    [[96, 96, 36], 
     [('c', [3, 1]), ('f', [5, False]), ('m', [2, 2])],
     [('t', [4, 2]),('c', [3, 1]),('i'),('f', [1, False])]], # third layer
    [[48, 48, 48], 
     [('c', [3, 1]), ('f', [3, False]), ('m', [2, 2])],
     [('t', [4, 2]),('c', [3, 1]),('i'),('f', [1, False])]], # forth layer
    [[24, 24, 64], 
     [('c', [3, 1]), ('f', [3, False])]], # fifth layer
    [[384, 384, 24], [('i'), ('c', [5, 1])]] # readout layer
    ]


class GammaNet(tf.keras.Model):
    '''
    Gamma-net class
    '''
    def __init__(self, batch_size=1, steps=1, blocks_config = default_config, mode='segmentation'):
        super().__init__()
        self.batch_size = batch_size
        self.n_layers = len(blocks_config) - 1
        self.steps = steps
        self.blocks_config = blocks_config
        self.blocks = [] # stores gammanetblocks, number of items equals number 
                         # of layers, each layer contains one or two blocks.
        self.mode = mode

        for i in range(self.n_layers + 1):
        # for all layers:
            block_config = self.blocks_config[i]
            input_shape = block_config[0]
            block = []
            for j in range(1, len(block_config)):
                if j == 1:
                    name = 'block_' + str(i) + '_ff'
                else: name = 'block_' + str(i) + '_fb'
                block.append(GammaNetBlock(self.batch_size, input_shape, block_config[j], name = name))
            self.blocks.append(block)

    def call(self, x):
        # pupulate a list that holds all the hidden states
        hidden_states = []
        for l in range(self.n_layers): # initalize the hidden states with the correct shapes
            hidden_states.append(tf.zeros(self.blocks[l][0].input_shape_))

        # run through the alrgoithm
        for i in range(self.steps):
            z = x 
            # In the paper, this assignment appears before the time loop,
            # and updates z on the first layer with ReLU and Conv every time 
            # step. 
            # This doesn't make much sense, because at time t, the input
            # to the first layer would already gone through t-1 ReLU and Convs,
            # but when you consider human brain, every second comes a fresh image
            # from the very bottom of the visual path.
            for l in range(self.n_layers):
                # bottom-up
                z, hidden_states[l] = self.blocks[l][0](z, hidden_states[l])
            if i == self.steps - 1:
                z_class = z
            
            for l in range(self.n_layers-2, -1, -1):
            # top-down
                z, hidden_states[l] = self.blocks[l][1](z, hidden_states[l])
        if self.mode == 'segmentation':
            out, _ = self.blocks[-1][0](z, None)
        elif self.mode == 'classification': 
            out, _ = self.blocks[-1][0](z_class, None)
        return out

################################################################################

# loss fuctions
def pixel_wise_bce(y_true, y_pred):
    # Flatten the prediction tensor and the true tensor
    y_pred_flat = tf.reshape(y_pred, [-1])
    y_true_flat = tf.reshape(y_true, [-1])

    # Compute binary cross entropy
    return tf.keras.losses.binary_crossentropy(y_true_flat, y_pred_flat)

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