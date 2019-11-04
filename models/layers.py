import tensorflow as tf
import keras
from keras import backend as K

def self_attention(x):
    n = int(x.get_shape()[1])
    attention = keras.layers.Dense(units=n, activation='sigmoid',
                                   kernel_initializer ='he_normal')(x)
    return keras.layers.merge.Multiply([x, attention])


def positional_squeeze_excitation(x, ratio):
    orgdim = int(x.get_shape()[1])
    return None
