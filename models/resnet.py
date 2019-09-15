import keras
import tensorflow as tf
from keras import backend as K

""" ResNet tf.keras """
class ResNetV2_keras():
    def __init__(self,training):
#         self.X_input = X_input
        self.training = training
        self.name = 'ResNetV2'
        self.in_filter = 16
        self.out_filter = None
        self.l2 = 1e-2
    
    def resnet_block(self,x, filter_size, kernels=3, stride=1 ,batch=True ,acti=True ):
        if batch:
#             x = tf.layers.batch_normalization(inputs = x, training=self.training)
            x = keras.layers.BatchNormalization(axis=-1)(x) #,training=self.training
        if acti:
            x = keras.layers.Activation('relu')(x)
        x = keras.layers.Conv2D(filters=filter_size,
                                   kernel_size=kernels,
                                   padding="SAME", strides=stride,
                                   kernel_initializer='he_normal',
                                   kernel_regularizer=tf.contrib.layers.l2_regularizer(self.l2)
                                  )(x)
        return x

    def build(self,input_shape) : #,X_input
        
        inputs = keras.layers.Input(input_shape)  
        x = keras.layers.Conv2D(filters=self.in_filter,
                                   kernel_size=3,
                                   padding="SAME", strides=1,
                                   kernel_initializer='he_normal',
                                   kernel_regularizer=tf.contrib.layers.l2_regularizer(self.l2)
                                  )(inputs)
#         x = tf.layers.batch_normalization(inputs = x, training=self.training)
        x = keras.layers.BatchNormalization(axis=-1)(x) #,training=self.training
        x = keras.layers.Activation('relu')(x)
        
        for stage in range(3):
            for blocks in range(12): 
                stride_ = 1
                if stage == 0:
                    ## 0: 16 -> 64
                    self.out_filter = self.in_filter *4 
                else:
                    ## 1: 62 -> 128 | 2: 128 -> 256
                    self.out_filter = self.in_filter *2 
                    if blocks == 0:
                        stride_ = 2 ## down size
                
                if blocks == 0:    
                    block_layer = self.resnet_block(x,self.in_filter, kernels=1
                                          , stride=stride_, batch = False, acti =False)
                    x = self.resnet_block(x,self.out_filter,kernels=1
                                          , stride=stride_, batch = False, acti =False)
                else:
                    block_layer = self.resnet_block(x,self.in_filter, kernels=1, batch = True, acti =True)

                block_layer = self.resnet_block(block_layer, self.in_filter, kernels=3, batch = True, acti =True)
                block_layer = self.resnet_block(block_layer, self.out_filter, kernels=1, batch = True, acti =True)

                x = keras.layers.Add()([block_layer, x])
            
            self.in_filter = self.out_filter
        
        
#         x = tf.layers.batch_normalization(inputs = x, training=self.training)
        x = keras.layers.BatchNormalization(axis=-1)(x) #,training=self.training
        x = keras.layers.Activation('relu')(x)
        x = keras.layers.AveragePooling2D(pool_size = 8)(x)
        x = keras.layers.Flatten()(x)
        out = keras.layers.Dense(units=1, activation='sigmoid',kernel_initializer ='he_normal')(x) 
        # out = tf.nn.sigmoid(x,name='output-layer')
        
        # keras model summary
        model = keras.Model(inputs=inputs, outputs=out)
        # print(model.summary())
        
        return model
    
    

""" ResNet tensorflow """
class ResNetV2():
    def __init__(self,training):
#         self.X_input = X_input
        self.training = training
        self.name = 'ResNetV2'
        self.in_filter = 16
        self.out_filter = None
    
    def resnet_block(self,x, filter_size, kernels=3, stride=1 ,batch=True ,acti=True ):
        if batch:
            x = tf.layers.batch_normalization(inputs = x, training=self.training)
#             x = tf.keras.layers.BatchNormalization()(x)
        if acti:
            x = tf.nn.relu(x)
        x = tf.layers.conv2d(inputs=x, filters=filter_size,
                               kernel_size=kernels,
                               padding="SAME", strides=stride,
                               kernel_initializer='he_normal',
                               kernel_regularizer=tf.contrib.layers.l2_regularizer(1e-4))
        return x

    def build(self,X_input):
#         inputs = tf.keras.layers.Input((32,32,3))  
        x = tf.layers.conv2d(inputs=X_input ,filters=self.in_filter,
                               kernel_size=3,
                               padding="SAME", strides=1,
                               kernel_initializer='he_normal',
                               kernel_regularizer=tf.contrib.layers.l2_regularizer(1e-4))
        x = tf.layers.batch_normalization(inputs = x, training=self.training)
#         x = tf.keras.layers.BatchNormalization()(x)
        x = tf.nn.relu(x)
        
        for stage in range(3):
            for blocks in range(12):
                stride_ = 1
                if stage == 0:
                    ## 0: 16 -> 64
                    self.out_filter = self.in_filter *4 
                else:
                    ## 1: 62 -> 128 | 2: 128 -> 256
                    self.out_filter = self.in_filter *2 
                    if blocks == 0:
                        stride_ = 2 ## down size
                
                if blocks == 0:    
                    block_layer = self.resnet_block(x,self.in_filter, kernels=1
                                          , stride=stride_, batch = False, acti =False)
                    x = self.resnet_block(x,self.out_filter,kernels=1
                                          , stride=stride_, batch = False, acti =False)
                else:
                    block_layer = self.resnet_block(x,self.in_filter, kernels=1, batch = True, acti =True)

                block_layer = self.resnet_block(block_layer, self.in_filter, kernels=3, batch = True, acti =True)
                block_layer = self.resnet_block(block_layer, self.out_filter, kernels=1, batch = True, acti =True)

                x = tf.keras.layers.Add()([block_layer, x])
            
            self.in_filter = self.out_filter
        
        
        x = tf.layers.batch_normalization(inputs = x, training=self.training)
#         x = tf.keras.layers.BatchNormalization()(x) #
        x = tf.nn.relu(x)
        x = tf.layers.average_pooling2d(x, pool_size=8,strides=8)
        x = tf.layers.flatten(x)
        x = tf.layers.dense(x, units=1) #, activation='sigmoid'
        out = tf.nn.sigmoid(x,name='output-layer')
        
        # keras model summary
#         model = tf.keras.Model(inputs=inputs, outputs=out)
#         print(model.summary())
        
        return out