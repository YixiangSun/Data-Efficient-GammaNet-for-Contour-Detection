import numpy as np
import tensorflow as tf
from tensorflow.python.keras import layers
from tensorflow.python.keras import initializers

class fGRUCell(tf.keras.layers.Layer):
    '''
    Generates an fGRUCell
    params:
    input_size: n x n
    hidden_channels: the number of channels which is constant throughout the
                     processing of each unit
    '''
    def __init__(self, input_size, hidden_channels, kernel_size=3, padding='same',
                 normtype='batchnorm', channel_sym=True, use_attention=0):
        super().__init__()
        self.input_size = input_size
        self.hidden_channels = hidden_channels
        self.kernel_size = kernel_size
        self.padding = padding
        self.normtype = normtype
        self.channel_sym = channel_sym

        if use_attention:
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

        # initiate other weights
        self.alpha = tf.Variable(0.1, dtype='float32')
        self.mu = tf.Variable(0, dtype='float32')
        self.nu = tf.Variable(0, dtype='float32')
        self.omega = tf.Variable(0.1, dtype='float32')
        self.delta = tf.Variable(np.zeros(self.hidden_channels)+0.1, dtype='float32')
        self.omicron = tf.Variable(np.zeros(self.hidden_channels), dtype='float32')
        self.eta = tf.Variable(np.random.rand(self.hidden_channels), dtype='float32')

    def channel_symmetrize(self):
        '''
        symmetrize the kernels channel-wise
        Somehow, if I write it in init, there will be the following error:
        'Conv2D' does not have attribute 'kernel'.
        '''
        if self.channel_sym: 
            for i in range(self.hidden_channels):
                for j in range(self.hidden_channels):
                    self.U_a.kernel[:,:,i,j].assign(self.U_a.kernel[:,:,j,i])
                    self.U_f.kernel[:,:,i,j].assign(self.U_f.kernel[:,:,j,i])
                    self.W_s.kernel[:,:,i,j].assign(self.W_s.kernel[:,:,j,i])
                    self.W_f.kernel[:,:,i,j].assign(self.W_f.kernel[:,:,j,i])
    
    def instance_norm(self, r):
        '''
        Param: r, a 4D tensor, b x h x w x c, where b = 1
        Return: a tensor normalized with the same size as r.
        '''                
        return np.array([self.omicron + self.delta * (r[0] - np.mean(r[0], axis=(0, 1)))\
                         /(np.sqrt(np.var(r[0], axis=(0, 1))+self.eta))])

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
        m_s_expanded = np.array([tf.transpose(m_s)[0]]*self.hidden_channels).T
        g_s = tf.sigmoid(self.instance_norm(a_s * m_s_expanded))
        # Compute suppression gate
        c_s = self.instance_norm(self.W_s(h * g_s))
        # compute suppression interactions
        S = tf.keras.activations.relu(z - tf.keras.activations.relu((self.alpha * h + self.mu)*c_s))
        # Additive and multiplicative suppression of Z

        # Stage 2: facilitation
        g_f = tf.sigmoid(self.instance_norm(self.U_f(S)))
        # Compute channel-wise recurrent updates
        c_f = self.instance_norm(self.W_f(S))
        # Compute facilitation interactions
        h_tilda = tf.keras.activations.relu(self.nu*(c_f + S) + self.omega*(c_f * S))
        # Additive and multiplicative facilitation of S
        ht = (1 - g_f) * h + g_f * h_tilda
        # Update recurrent state
        return ht