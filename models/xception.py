import keras
import tensorflow as tf
from keras import backend as K

""" Xception """
class Xception():
    def __init__(self):
        self.l2 = 1e-4
    
    def conv_block(self,x, f, k, s, pad, acti =True, batch=True):
        x = keras.layers.Conv2D(filters=f, kernel_size=k, strides=s,
                                padding=pad, activation = 'linear',
                                data_format = "channels_last",
                                kernel_regularizer=keras.regularizers.l2(self.l2),
                                kernel_initializer='he_normal')(x)
        if batch:
            x = keras.layers.BatchNormalization(axis=-1)(x)
        if acti :
            x = keras.layers.Activation('relu')(x)
        return x
    
    def separable_conv_block(self,x, f, k, s, pad, acti =True, batch=True, dataformat = "channels_last"):
        bn_axis = 1 if dataformat == "channels_first" else -1
        if acti :
            x = keras.layers.Activation('relu')(x)

        x = keras.layers.SeparableConv2D(filters=f, kernel_size=k, strides=s,
                                         padding=pad, activation = 'linear',
                                         data_format = dataformat,
                                         kernel_regularizer=keras.regularizers.l2(self.l2),
                                         kernel_initializer='he_normal')(x)
        if batch:
            x = keras.layers.BatchNormalization(axis=bn_axis)(x)
        return x
    
    def pooling(self, x, pool, stride, pad='valid', types="MAX"):
        if types == "MAX":
            return keras.layers.MaxPooling2D(pool_size=pool, strides=stride, padding=pad)(x)
        elif types == "AVG":
            return keras.layers.AveragePooling2D(pool_size=pool, strides=stride, padding=pad)(x)
        else:
            raise ValueError('invalied pooling type')
    
    def build(self, input_shape=(299,299,3)):
        inputs = keras.layers.Input(input_shape)
        x = self.conv_block(inputs,32, (3,3), (2,2), 'valid', acti =True, batch=True)
        x = self.conv_block(x     ,64, (3,3), (1,1), 'valid', acti =True, batch=True)
        
        ## Entry flow 1
        x1 = self.conv_block(x,128, (1,1), (2,2), 'valid', acti =False, batch=True)
        x2 = self.separable_conv_block(x, 128, (3,3), (1,1), 'same', acti =False, batch=True, dataformat = "channels_last")
        x2 = self.separable_conv_block(x2, 128, (3,3), (1,1), 'same', acti =True, batch=True, dataformat = "channels_last")
        x2 = self.pooling(x2, (3,3), (2,2), pad='same', types="MAX")
        x = keras.layers.Add()([x1, x2])
        
        ## Entry flow 2
        x1 = self.conv_block(x,256, (1,1), (2,2), 'valid', acti =False, batch=True)
        x2 = self.separable_conv_block(x, 256, (3,3), (1,1), 'same', acti =True, batch=True, dataformat = "channels_last")
        x2 = self.separable_conv_block(x2, 256, (3,3), (1,1), 'same', acti =True, batch=True, dataformat = "channels_last")
        x2 = self.pooling(x2, (3,3), (2,2), pad='same', types="MAX")
        x = keras.layers.Add()([x1, x2])
        
        ## Entry flow 3
        x1 = self.conv_block(x,728, (1,1), (2,2), 'valid', acti =False, batch=True)
        x2 = self.separable_conv_block(x, 728, (3,3), (1,1), 'same', acti =True, batch=True, dataformat = "channels_last")
        x2 = self.separable_conv_block(x2, 728, (3,3), (1,1), 'same', acti =True, batch=True, dataformat = "channels_last")
        x2 = self.pooling(x2, (3,3), (2,2), pad='same', types="MAX")
        x = keras.layers.Add()([x1, x2])
        
        ## Middle flow 
        for i in range(8):
            x2 = self.separable_conv_block(x, 728, (3,3), (1,1), 'same', acti =True, batch=True, dataformat = "channels_last")
            x2 = self.separable_conv_block(x2, 728, (3,3), (1,1), 'same', acti =True, batch=True, dataformat = "channels_last")
            x2 = self.separable_conv_block(x2, 728, (3,3), (1,1), 'same', acti =True, batch=True, dataformat = "channels_last")
            x = keras.layers.Add()([x, x2])
        
        ## Exit flow 1
        x1 = self.conv_block(x,1024, (1,1), (2,2), 'valid', acti =False, batch=True)
        x2 = self.separable_conv_block(x, 728, (3,3), (1,1), 'same', acti =True, batch=True, dataformat = "channels_last")
        x2 = self.separable_conv_block(x2, 1024, (3,3), (1,1), 'same', acti =True, batch=True, dataformat = "channels_last")
        x2 = self.pooling(x2, (3,3), (2,2), pad='same', types="MAX")
        x = keras.layers.Add()([x1, x2])
        
        ## Exit flow 2
        x = self.separable_conv_block(x, 1536, (3,3), (1,1), 'same', acti =False, batch=True, dataformat = "channels_last")
        x = keras.layers.Activation('relu')(x)
        x = self.separable_conv_block(x, 2048, (3,3), (1,1), 'same', acti =False, batch=True, dataformat = "channels_last")
        x = keras.layers.Activation('relu')(x)
        
        x = keras.layers.GlobalAveragePooling2D(data_format="channels_last")(x)
        
        ## Fully connnected 
        out = keras.layers.Dense(units=1, activation='sigmoid',kernel_initializer ='he_normal')(x)  #'sigmoid'
        
        return keras.Model(inputs=inputs, outputs=out)