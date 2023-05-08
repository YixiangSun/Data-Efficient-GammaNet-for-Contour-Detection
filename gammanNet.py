import numpy as np
import tensorflow as tf
from tensorflow.python.keras import layers
import fGRU

class GammaNetBlock(layers.Layer):
    '''
    Generate a block in gamma-net
    '''
    def __init__(self, batch_size, input_shape, layers_config):
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
        '''
        super().__init__()
        self.batch_size = batch_size

        self.input_shape_ = [batch_size, input_shape[0], input_shape[1], input_shape[2]]
        # here, input_shape_ is in [batch_size, height, width, channel_size]

        self.hidden_channels = self.input_shape_[-1]
        self.fgru = None
        self.hidden_state = None
        self.layers_config = layers_config
        self.layers = []

        for layer in layers_config:
        # populate the blocks with layers
            if layer[0] == 'c':
                kernel_size = layer[1][0]
                strides = layer[1][1]
                self.layers.append(layers.Conv2D(
                    filters=self.hidden_channels, 
                    kernel_size=kernel_size, 
                    strides=strides,
                    padding='same',
                    activation='ReLU'
                    ))
                
            elif layer[0] == 't':
                kernel_size = layer[1][0]
                strides = layer[1][1]
                self.layers.append(layers.Conv2DTranspose(
                    filters=self.hidden_channels, 
                    kernel_size=kernel_size, 
                    strides=strides,
                    padding='same' 
                    ))
                
            elif layer[0] == 'f':
                kernel_size = layer[1][0]
                use_attention = layer[1][1]
                self.fgru=fGRU.fGRU(
                    input_shape=self.input_shape_, 
                    kernel_size=kernel_size,
                    use_attention = use_attention,
                    )
                self.layers.append(self.fgru)

            elif layer[0] == 'm':
                pool_size = layer[1][0]
                strides = layer[1][1]
                self.layers.append(layers.MaxPool2D(
                    pool_size=pool_size, 
                    strides=strides,
                    padding='valid' 
                    ))
                
            elif layer[0] == 'i':
                self.layers.append(fGRU.InstanceNorm(self.hidden_channels))

    def call(self, x, h):
        z = x
        for layer in self.layers:
            if layer == self.fgru:
                print('successfully went to fgru') # for debugging
                z = layer(z, h)
                print('successfully got the output of fgru') # for debugging
                self.hidden_state = z
            else:
                z = layer(z)
        return z

# in each layer, there are three lists: 
# input shape (without batch_size), bottom-up unit, top-down unit (if any)
default_config = [
    [[384, 384, 24], 
     [('c', [3, 1]), ('c', [3, 1]), ('f', [9, False]), ('m', [2, 2])],
     [('t', [4, 2]),('c', [3, 1]),('i'),('f', [1, False])]], # first layer
    [[192, 192, 28], 
     [('c', [3, 1]), ('f', [7, False]), ('m', [2, 2])],
     [('t', [4, 2]),('c', [3, 1]),('i'),('f', [1, False])]], # second layer
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
    def __init__(self, batch_size=1, steps=1, blocks_config = default_config):
        super().__init__()
        self.batch_size = batch_size
        self.n_layers = len(blocks_config) - 1
        self.steps = steps
        self.blocks_config = blocks_config
        self.blocks = [] # stores gammanetblocks, number of items equals number 
                         # of layers, each layer contains one or two blocks.

        for i in range(self.n_layers + 1):
        # for all layers:
            block_config = self.blocks_config[i]
            input_shape = block_config[0]
            block = []
            for j in range(1, len(block_config)):
                block.append(GammaNetBlock(self.batch_size, input_shape, block_config[j]))
            self.blocks.append(block)

    def call(self, x):
        for _ in range(self.steps):
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
                if l == self.n_layers-1:
                    h = self.blocks[l][0].hidden_state
                else: h = self.blocks[l][1].hidden_state
                if h == None:
                # if no initial hidden_state, assign h as 0.
                # note that the input_shape_ of gammaNetBlock objects contains
                # batch_size aat the begginning already.
                    h = tf.zeros(self.blocks[l][0].input_shape_)
                z = self.blocks[l][0](z, h)
            
            for l in range(self.n_layers-2, -1, -1):
            # top-down
                h = self.blocks[l][0].hidden_state
                z = self.blocks[l][1](z, h)
        
        out = self.blocks[-1][0](z, None)
        print('went to final output') # for debugging
        return out