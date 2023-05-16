import numpy as np
import tensorflow as tf
from tensorflow.python.keras import layers
from tensorflow.python.keras import initializers

class fGRU(tf.keras.layers.Layer):
    '''
    Generates an fGRUCell
    params:
    hidden_channels: the number of channels which is constant throughout the
                     processing of each unit
    '''
    def __init__(self, input_shape, kernel_size=3, padding='same', use_attention=0, channel_sym = True, name = 'fGRU'):
        # channel_sym assigned False for speed. Saves 30 seconds.

        super().__init__(name = name)
        self.hidden_channels = input_shape[-1]
        self.kernel_size = kernel_size
        self.padding = padding
        self.channel_sym = channel_sym
        self.use_attention = use_attention
        self.input_shape_ = input_shape

        if self.use_attention:
            # TODO: implement attention
            pass
        else:
            # Initialize convolutional kernels
            self.U_a = layers.Conv2D(
                filters=self.hidden_channels,
                kernel_size=1, 
                strides=1, 
                padding=self.padding,
                kernel_initializer=initializers.Orthogonal(),
                )
            
            self.U_m = layers.Conv2D(
                filters=1,
                kernel_size=self.kernel_size, 
                strides=1, 
                padding=self.padding,
                kernel_initializer=initializers.Orthogonal(),
                )
            
            self.W_s = layers.Conv2D(
                filters=self.hidden_channels,
                kernel_size=self.kernel_size, 
                strides=1, 
                padding=self.padding,
                kernel_initializer=initializers.Orthogonal(),
                )
            
            self.U_f = layers.Conv2D(
                filters=self.hidden_channels,
                kernel_size=self.kernel_size, 
                strides=1, 
                padding=self.padding,
                kernel_initializer=initializers.Orthogonal(),
                )
            
            self.W_f = layers.Conv2D(
                filters=self.hidden_channels,
                kernel_size=self.kernel_size, 
                strides=1, 
                padding=self.padding,
                kernel_initializer=initializers.Orthogonal(),
                )
        self.build(self.input_shape_)

        # initiate other weights
        self.alpha = tf.Variable(0.1, dtype='float32')
        self.mu = tf.Variable(0, dtype='float32')
        self.nu = tf.Variable(0, dtype='float32')
        self.omega = tf.Variable(0.1, dtype='float32')

    def channel_symmetrize(self):
        '''
        symmetrize the kernels channel-wise
        Somehow, if I write it in init, there will be the following error:
        'Conv2D' does not have attribute 'kernel'.
        '''
        if self.channel_sym: 
            for i in range(self.hidden_channels):
                for j in range(i, self.hidden_channels):
                    self.U_a.kernel[:,:,i,j].assign(self.U_a.kernel[:,:,j,i])
                    self.U_f.kernel[:,:,i,j].assign(self.U_f.kernel[:,:,j,i])
                    self.W_s.kernel[:,:,i,j].assign(self.W_s.kernel[:,:,j,i])
                    self.W_f.kernel[:,:,i,j].assign(self.W_f.kernel[:,:,j,i])

    def build(self, input_shape):
        self.U_a.build(input_shape)
        self.U_m.build(input_shape)
        self.U_f.build(input_shape)
        self.W_s.build(input_shape)
        self.W_f.build(input_shape)
        if self.channel_sym:
            self.channel_symmetrize()
        
        # initialize instance norm layers
        self.iN1 = InstanceNorm(self.hidden_channels)
        self.iN2 = InstanceNorm(self.hidden_channels)
        self.iN3 = InstanceNorm(self.hidden_channels)
        self.iN4 = InstanceNorm(self.hidden_channels)


    def call(self, z, h):
        '''
        Params: 
        Z: output from the last layer if fGRU-horizontal, hidden state of the
        current layer at t if fGRU-feedback.
        H: hidden state of the current layer at t-1 if fGRU-horizontal, output
        from the next layer if fGRU-feedback.
        '''

        # Stage 1: suppression
        a_s = self.U_a(h) # Compute channel-wise selection
        m_s = self.U_m(h) # Compute spatial selection
        # (note that U_a and U_m are kernels of different sizes and therefore
        # have different functions)

        m_s_expanded = tf.transpose(tf.convert_to_tensor([tf.transpose(m_s)[0]]*self.hidden_channels))
        g_s = tf.sigmoid(self.iN1(a_s * m_s_expanded))
        # Compute suppression gate
        c_s = self.iN2(self.W_s(h * g_s))
        # compute suppression interactions
        S = tf.keras.activations.relu(z - tf.keras.activations.relu((self.alpha * h + self.mu)*c_s))
        # Additive and multiplicative suppression of Z

        # Stage 2: facilitation
        g_f = tf.sigmoid(self.iN3(self.U_f(S)))
        # Compute channel-wise recurrent updates
        c_f = self.iN4(self.W_f(S))
        # Compute facilitation interactions
        h_tilda = tf.keras.activations.relu(self.nu*(c_f + S) + self.omega*(c_f * S))
        # Additive and multiplicative facilitation of S
        ht = (1 - g_f) * h + g_f * h_tilda
        # Update recurrent state
        return ht

class InstanceNorm(tf.keras.layers.Layer):
    def __init__(self, hidden_channels, name = 'instance_norm'):
        super().__init__(name = name)
        self.hidden_channels = hidden_channels
        self.omicron = tf.Variable(np.zeros(self.hidden_channels), dtype='float32')
        self.eta = tf.Variable(np.random.rand(self.hidden_channels), dtype='float32')
        self.delta = tf.Variable(np.zeros(self.hidden_channels)+0.1, dtype='float32')

    def call(self, r):
        '''
        Param: r, a 4D tensor, b x h x w x c, where b = 1
        Return: a tensor normalized with the same size as r.
        '''                
        return self.omicron + self.delta * (r - tf.math.reduce_mean(r, axis = (1, 2), keepdims = True))\
                        / tf.math.sqrt(tf.math.reduce_variance(r, axis = (1, 2), keepdims = True) + self.eta)