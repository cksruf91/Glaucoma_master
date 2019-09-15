import keras
import tensorflow as tf
from keras import backend as K

""" Unet  """    
# > https://github.com/jskDr/keraspp/blob/master/ex8_1_unet_cifar10.py
# It consists of the repeated application of two 3x3 convolutions (unpadded convolutions)
# each followed by a rectified linear unit (ReLU) and a 2x2 max pooling operation with stride 2
# At each downsampling step we double the number of feature channels

# upsampling of the feature map followed by a 2x2 convolution (“up-convolution”)
# that halves the number of feature channels, a concatenation with the correspondingly cropped feature map from the contracting path
# two 3x3 convolutions, each followed by a ReLU
# The cropping is necessary due to the loss of border pixels in every convolution

# At the final layer a 1x1 convolution is used to map each 64-component feature vector to the desired number of classes.
#  In total the network has 23 convolutional layers for downsampling
class Unet():
    def __init__(self):
        self.name = 'Unet'
        self.filter_size = 64
        self.l2 = 1e-6
        
        if K.image_data_format() != 'channels_last':
            raise ValueError('channels first')
    
    def conv_block(self, x, fsize, pad="same" ,k_size=3, s_size=1, acti=True,batch=True):
        x = keras.layers.Conv2D(filters=fsize, kernel_size=k_size,
                                padding=pad, strides=s_size,
                                activation = 'linear',
                                kernel_regularizer=keras.regularizers.l2(self.l2),
                                kernel_initializer='he_normal')(x)
        if batch:
            x = keras.layers.BatchNormalization(axis=-1)(x)
        if acti:
            x = keras.layers.Activation('relu')(x)
        return x
    
    def pooling(self, x, pool, stride, pad='valid', types="MAX"):
        if types == "MAX":
            return keras.layers.MaxPooling2D(pool_size=pool, strides=stride, padding=pad)(x)
        elif types == "AVG":
            return keras.layers.AveragePooling2D(pool_size=pool, strides=stride, padding=pad)(x)
        else:
            raise ValueError('invalied pooling type')
            

    def upconv(self, x, fsize, pad='valid', k_size=2, s_size=2, acti=True, batch=True):
        x = keras.layers.Conv2DTranspose(filters=fsize, kernel_size =k_size, strides =s_size, padding=pad
                                     ,kernel_regularizer=keras.regularizers.l2(self.l2))(x)
        if batch:
            x = keras.layers.BatchNormalization(axis=-1)(x)
        if acti:
            x = keras.layers.Activation('relu')(x)
        return x
            
    def build(self, input_shape):
        inputs = keras.layers.Input(input_shape)
        
        fs = self.filter_size # 64
        x = self.conv_block(inputs, fs, pad='same', acti=True, batch=True)
        c1 = self.conv_block(x, fs, pad='same', acti=True, batch=True)
        #c1 = keras.layers.Dropout(rate=0.3)(x)
        x = self.pooling(c1,2,2,pad='same', types="MAX")
        
        fs = int(fs*2) # 128
        x = self.conv_block(x, fs, pad='same', acti=True, batch=True)
        c2 = self.conv_block(x, fs, pad='same', acti=True, batch=True)
        #c2 = keras.layers.Dropout(rate=0.3)(x)
        x = self.pooling(c2,2,2,pad='same', types="MAX")
        
        fs = int(fs*2) # 256
        x = self.conv_block(x, fs, pad='same', acti=True, batch=True)
        c3 = self.conv_block(x, fs, pad='same', acti=True, batch=True)
        #c3 = keras.layers.Dropout(rate=0.3)(x)
        x = self.pooling(c3,2,2,pad='same', types="MAX")
        
        fs = int(fs*2) # 512
        x = self.conv_block(x, fs, pad='same', acti=True, batch=True)
        c4 = self.conv_block(x, fs, pad='same', acti=True, batch=True)
#         c4 = keras.layers.Dropout(rate=0.3)(c4)
        x = self.pooling(c4,2,2,pad='same', types="MAX")
        
        fs = int(fs*2) # 1024
        x = self.conv_block(x, fs, pad='same', acti=True, batch=True)
        x = self.conv_block(x, fs, pad='same', acti=True, batch=True)
#         x = keras.layers.Dropout(rate=0.3)(x)
        #x = self.encode(x, [fs,fs], pad='same', pool=False, drop =True)
        
        fs = int(fs/2)
        x = self.upconv(x, fs, k_size=3, s_size=2, pad='same', acti=False, batch=False) 
        x = keras.layers.Concatenate(axis=-1)([x, c4])
        #x = keras.layers.Dropout(rate=0.3)(x)
        x = self.conv_block(x, fs, pad="same", acti=True,batch=True)
        x = self.conv_block(x, fs, pad="same", acti=True,batch=True)
        #x = self.upconv(x, fs, pad='same', k_size=3, s_size=1, acti=True, batch=True)
        #x = self.upconv(x, fs, pad='same', k_size=3, s_size=1, acti=True, batch=True)
        
        fs = int(fs/2)
        x = self.upconv(x, fs, k_size=3, s_size=2, pad='same', acti=False, batch=False) 
        x = keras.layers.Concatenate(axis=-1)([x, c3])
        #x = keras.layers.Dropout(rate=0.3)(x)
        x = self.conv_block(x, fs, pad="same", acti=True,batch=True)
        x = self.conv_block(x, fs, pad="same", acti=True,batch=True)
        
        fs = int(fs/2)
        x = self.upconv(x, fs, k_size=3, s_size=2, pad='same', acti=False, batch=False) 
        x = keras.layers.Concatenate(axis=-1)([x, c2])
        #x = keras.layers.Dropout(rate=0.3)(x)
        x = self.conv_block(x, fs, pad="same", acti=True,batch=True)
        x = self.conv_block(x, fs, pad="same", acti=True,batch=True)
        
        fs = int(fs/2)
        x = self.upconv(x, fs, k_size=3, s_size=2, pad='same', acti=False, batch=False) 
        x = keras.layers.Concatenate(axis=-1)([x, c1])
        #x = keras.layers.Dropout(rate=0.3)(x)
        x = self.conv_block(x, fs, pad="same", acti=True,batch=True)
        x = self.conv_block(x, fs, pad="same", acti=True,batch=True)
        
        x = self.conv_block(x, 2, pad='same', k_size=1, s_size=1, acti=False,batch=False)
        out = keras.layers.Activation('softmax')(x) #sigmoid

        return keras.Model(inputs=inputs, outputs=out)