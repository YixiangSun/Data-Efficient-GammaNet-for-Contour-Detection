from types import SimpleNamespace

import numpy as np
import tensorflow as tf

## Run functions eagerly to allow numpy conversions.
## Enable experimental debug mode to suppress warning (feel free to remove second line)
tf.config.run_functions_eagerly(True)
tf.data.experimental.enable_debug_mode()


###############################################################################################


def get_data():
    """
    Loads CIFAR10 training and testing datasets

    :return X0: training images,
            Y0: training labels,
            X1: testing images,
            Y1: testing labels
            D0: TF Dataset training subset
            D1: TF Dataset testing subset
        D_info: TF Dataset metadata
    """

    ## This process may take a bit to load the first time; should get much faster
    import tensorflow_datasets as tfds

    ## Overview of dataset downloading: https://www.tensorflow.org/datasets/catalog/overview
    ## CIFAR-10 Dataset https://www.tensorflow.org/datasets/catalog/cifar10
    (D0, D1), D_info = tfds.load(
        "cifar10", as_supervised=True, split=["train[:50%]", "test"], with_info=True
    )

    X0, X1 = [np.array([r[0] for r in tfds.as_numpy(D)]) for D in (D0, D1)]
    Y0, Y1 = [np.array([r[1] for r in tfds.as_numpy(D)]) for D in (D0, D1)]

    return X0, Y0, X1, Y1, D0, D1, D_info

###############################################################################################

import fGRU
import gammaNet
data = get_data()

classification_config = [
    [[32, 32, 3], 
     [('c', [3, 1]), ('c', [3, 1]), ('f', [9, False]), ('m', [2, 2])],
     [('t', [4, 2]),('c', [3, 1]),('i'),('f', [1, False])]], # first layer
    [[16, 16, 5], 
     [('c', [3, 1]), ('f', [7, False]), ('m', [2, 2])],
     [('t', [4, 2]),('c', [3, 1]),('i'),('f', [1, False])]], # second layer
    [[8, 8, 10], 
     [('c', [3, 1]), ('f', [5, False]), ('m', [2, 2])],
     [('t', [4, 2]),('c', [3, 1]),('i'),('f', [1, False])]], # third layer
    [[4, 4, 20], 
     [('c', [3, 1]), ('f', [3, False])],], # forth layer # fifth layer
    [[4, 4, 20], [('i'), ('flat'), ('c', [5, 1])]] # readout layer
    ]

X0, Y0, X1, Y1, D0, D1, D_info = data

model = gammaNet(steps = 4, blocks_config = classification_config, mode='classification')
model.build()
model.compile(optimizer=tf.optimizers.Adam(), loss=tf.keras.losses.CategoricalCrossentropy)

model.fit(X0, Y0)