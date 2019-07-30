import keras
# from keras.layers import Input, regularizers, Conv1D , MaxPooling1D, Flatten, Dense, Conv2D, MaxPooling2D, Dropout
# from keras.models import Sequential
from sklearn.utils import class_weight
# from keras.regularizers import l2
import tensorflow as tf

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
    def __init__(self,is_train):
        self.is_train = is_train
        self.name = 'Unet'
        self.filter_size = 64
    
    def conv_block(self, x, fsize, pad="same" ,k_size=3, s_size=1, acti=True,batch=True):
        x = keras.layers.Conv2D(filters=fsize, kernel_size=k_size,
                                padding=pad, strides=s_size,
                                kernel_regularizer=keras.regularizers.l2(1e-4),
                                kernel_initializer='he_normal')(x)
        #x = tf.layers.conv2d(inputs=x, filters=fsize,
        #                     kernel_size=k_size,
        #                     padding=pad, strides=s_size,
        #                     kernel_regularizer=tf.contrib.layers.l2_regularizer(1e-4),
        #                     kernel_initializer='he_normal',
        #                     )
        if batch:
            #x = tf.layers.batch_normalization(inputs = x, training=self.is_train)
            x = keras.layers.BatchNormalization()(x , training=self.training)
        if acti:
            #x = tf.nn.relu(x)
            x = keras.layers.Activation('relu')(x)
        return x
    
    def pooling(self, x, pool, stride, pad='valid', types="MAX"):
        if types == "MAX":
            # tf.layers.max_pooling2d(x, pool_size=pool, strides=stride, padding=pad)
            return keras.layers.MaxPooling2D(pool_size=pool, strides=stride, padding=pad)(x)
        elif types == "AVG":
            # tf.layers.average_pooling2d(x, pool_size=pool, strides=stride, padding=pad)
            return keras.layers.AveragePooling2D(pool_size=pool, strides=stride, padding=pad)(x)
        else:
            raise ValueError('invalied pooling type')
    
    def encode(self,x, f_size, pad='valid', pool=False):
        x = self.conv_block(x, f_size[0], pad="same" ,acti=True,batch=True)
        x = self.conv_block(x, f_size[1], pad="same" ,acti=True,batch=True)
        if pool:
            p = self.pooling(x,2,2,pad=pad, types="MAX")
            return x ,p
        
        else:
            return x
            

    def upconv_concat(self, x, b , fsize, pad='valid', k_size=2, s_size=2):
        x = keras.layers.Conv2DTranspose(filters=fsize, kernel_size =k_size, strides =s_size, padding=pad
                                     ,kernel_regularizer=keras.regularizers.l2(1e-4))(x)
        # x = tf.layers.conv2d_transpose(x ,filters=fsize,
        #                                kernel_size =k_size, strides =s_size,
        #                                padding=pad,
        #                                kernel_regularizer=tf.contrib.layers.l2_regularizer(1e-4))
        # tf.concat([x, b], axis=-1)
        return keras.layers.Concatenate([x, b] , axis=-1)
            
    def build(self, x):
        inputs = keras.layers.Input(input_shape)
        
        fs = self.filter_size
        bridge1, x = self.encode(inputs, [fs,fs],pad='same',pool=True)
        fs *= 2
        bridge2, x = self.encode(x, [fs,fs],pad='same',pool=True)
        fs *= 2
        bridge3, x = self.encode(x, [fs,fs],pad='same',pool=True)
        fs *= 2
        bridge4, x = self.encode(x, [fs,fs],pad='same',pool=True)
        fs *= 2
        
        x = self.encode(x, [fs,fs],pool=False, pad='same')
        # print(x.get_shape().as_list())
        
        fs /= 2
        x = self.upconv_concat(x, bridge4, 256, pad='same')
        x = self.encode(x, [fs,fs],pool=False, pad='same')
        
        fs /= 2
        x = self.upconv_concat(x, bridge3, 128, pad='same')
        x = self.encode(x, [fs,fs],pool=False, pad='same')
        
        fs /= 2
        x = self.upconv_concat(x, bridge2, 64, pad='same')
        x = self.encode(x, [fs,fs],pool=False, pad='same')
        
        fs /= 2
        x = self.upconv_concat(x, bridge1, 32, pad='same')
        x = self.encode(x, [fs,fs],pool=False, pad='same')
        
        x = self.conv_block(x, 1, k_size=1, s_size=1, acti=False,batch=False, pad='same')
        # tf.nn.sigmoid(x)
        out = keras.activations.sigmoid(x)
        
        model = keras.Model(inputs=inputs, outputs=out)
        
        return model


""" Inception v4  """
class InceptionV4():
    def __init__(self,is_train):
        self.is_train = is_train
        self.name = 'InceptionV4'
        self.k = 192
        self.i = 224
        self.m = 256
        self.n = 384
    
    def conv_block(self,x,ksize,fsize,stride,pad="valid",acti=True,batch=False):
        x = tf.layers.conv2d(inputs=x, filters=fsize,
                             kernel_size=ksize,
                             padding=pad, strides=stride,
                             kernel_initializer='he_normal',
                             )
        # kernel_regularizer=tf.contrib.layers.l2_regularizer(1e-4)
        if batch:
            x = tf.layers.batch_normalization(inputs = x, training=self.is_train)
        if acti:
            x = tf.nn.relu(x)
        return x
    
    def pooling(self, x, pool, stride, pad, types="MAX"):
        if types == "MAX":
            return tf.layers.max_pooling2d(x, pool_size=pool, strides=stride, padding=pad)
        elif types == "AVG":
            return tf.layers.average_pooling2d(x, pool_size=pool, strides=stride, padding=pad)
        else:
            raise ValueError('invalied pooling type')
    
    def stem_block(self,x):
        x = self.conv_block(x,3,32,2,"valid")
        x = self.conv_block(x,3,32,1,"valid")
        x = self.conv_block(x,3,64,1,"same")
        
        node1 = self.pooling(x,3,2,"valid", types="MAX")
        node2 = self.conv_block(x,3,64,2,"valid")
        
        x = tf.concat((node1, node2),axis=-1) # stem_block1 73x73x160
        
        node1 = self.conv_block(x,1,64,1,"same")
        node1 = self.conv_block(node1,(7,1),64,1,"same")
        node1 = self.conv_block(node1,(1,7),64,1,"same")
        node1 = self.conv_block(node1,3,96,1,"valid")
        
        node2 = self.conv_block(x,1,64,1,"same")
        node2 = self.conv_block(node2,3,96,1,"valid")
        
        x = tf.concat((node1, node2),axis=-1) # stem_block2 71x71x192
        
        node1 = self.pooling(x,3,2,"valid", types="MAX")
        node2 = self.conv_block(x,3,192,2,"valid")
        
        x = tf.concat((node1, node2),axis=-1) # stem_block3 71x71x192
        
        return x
    
    def inception_A(self,x):
#         if types.upper() not in ["A","B","C"]:
#             raise ValueError('invalied value : inceoption layer type should be one of a,A,b,B,c,C')
#         if types = "A":
        node1 = self.conv_block(x,1,96,1,"same")
        node1 = self.pooling(node1,3,1,"same",types="AVG")

        node2 = self.conv_block(x,1,96,1,"same")

        node3 = self.conv_block(x,3,96,1,"same")
        node3 = self.conv_block(node3,1,64,1,"same")

        node4 = self.conv_block(x,3,96,1,"same")
        node4 = self.conv_block(node4,3,96,1,"same")
        node4 = self.conv_block(node4,1,64,1,"same")

        return tf.concat((node1, node2, node3, node4),axis=-1) # 71x71x192
    
    def inception_B(self,x):
        
        node1 = self.conv_block(x,1,128,1,"same")
        node1 = self.pooling(node1,3,1,"same",types="AVG")

        node2 = self.conv_block(x,1,364,1,"same")

        node3 = self.conv_block(x,(7,1),256,1,"same")
        node3 = self.conv_block(node3,(1,7),224,1,"same")
        node3 = self.conv_block(node3,1,192,1,"same")

        node4 = self.conv_block(x,(7,1),256,1,"same")
        node4 = self.conv_block(node4,(1,7),224,1,"same")
        node4 = self.conv_block(node4,(7,1),224,1,"same")
        node4 = self.conv_block(node4,(1,7),192,1,"same")
        node4 = self.conv_block(node4,1,192,1,"same")

        return tf.concat((node1, node2, node3, node4),axis=-1) # 71x71x192
            
    def inception_C(self,x):
        node1 = self.conv_block(x,1,256,1,"same")
        node1 = self.pooling(node1,3,1,"same",types="AVG")

        node2 = self.conv_block(x,1,256,1,"same")

        node3_1 = self.conv_block(x,(3,1),256,1,"same")
        node3_2 = self.conv_block(x,(1,3),256,1,"same")
        node3 = tf.concat((node3_1, node3_2), axis =-1)
        node3 = self.conv_block(node3,1,384,1,"same")

        node4_1 = self.conv_block(x,(3,1),256,1,"same")
        node4_2 = self.conv_block(x,(1,3),256,1,"same")
        node4 = tf.concat((node4_1, node4_2), axis =-1)
        node4 = self.conv_block(node4,(3,1),512,1,"same")
        node4 = self.conv_block(node4,(1,3),448,1,"same")
        node4 = self.conv_block(node4,1,384,1,"same")

        return tf.concat((node1, node2, node3, node4),axis=-1) # 71x71x192
        
    def reduction_A(self,x):
#         if types.upper() not in ["A","B"]:
#             raise ValueError('invalied value : reduction layer type should be a,A or b,B')
        node1 = self.pooling(x, 3, 2, "valid",types="MAX")
        
        node2 = self.conv_block(x,3,self.n,2,"valid")
        
        node3 = self.conv_block(x, 3, self.m, 2, "valid")
        node3 = self.conv_block(node3, 3, self.i, 1, "same")
        node3 = self.conv_block(node3, 1, self.k, 1, "same")
        
        return tf.concat((node1, node2, node3),axis=-1) # 71x71x192
        
    def reduction_B(self,x):
#         if types.upper() not in ["A","B"]:
#             raise ValueError('invalied value : reduction layer type should be a,A or b,B')
        node1 = self.pooling(x,3,2,"valid",types="MAX")
        
        node2 = self.conv_block(x,3,192,2,"valid")
        node2 = self.conv_block(node2,1,192,1,"same")
        
        node3 = self.conv_block(x,3,320,2,"valid")
        node3 = self.conv_block(node3,(7,1),320,1,"same")
        node3 = self.conv_block(node3,(1,7),256,1,"same")
        node3 = self.conv_block(node3,1,256,1,"same")
        
        return tf.concat((node1, node2, node3),axis=-1) # 71x71x192

    
    def fullyconneted(self,x,is_train):
        x = self.pooling(x,3,2,"valid",types="AVG")
        x = tf.layers.flatten(x)
        x = tf.layers.dropout(x,rate=0.2,training=is_train)
        x = tf.layers.dense(x, units=1) #, activation='sigmoid'
        return x

    def build(self,X_input) :
        x = self.stem_block(X_input)
        for i in range(7):
            x = self.inception_A(x)
        x = self.reduction_A(x)
        for i in range(7):
            x = self.inception_B(x)
        x = self.reduction_B(x)
        for i in range(7):
            x = self.inception_C(x)
        x = self.fullyconneted(x, is_train=self.is_train)
        out = tf.nn.sigmoid(x)
        return out




""" ResNet tf.keras """
class ResNetV3():
    def __init__(self,training):
#         self.X_input = X_input
        self.training = training
        self.name = 'ResNetV2'
        self.in_filter = 16
        self.out_filter = None
    
    def resnet_block(self,x, filter_size, kernels=3, stride=1 ,batch=True ,acti=True ):
        if batch:
#             x = tf.layers.batch_normalization(inputs = x, training=self.training)
            x = keras.layers.BatchNormalization()(x,training=self.training)
        if acti:
            x = keras.layers.Activation('relu')(x)
        x = keras.layers.Conv2D(filters=filter_size,
                                   kernel_size=kernels,
                                   padding="SAME", strides=stride,
                                   kernel_initializer='he_normal',
                                   kernel_regularizer=tf.contrib.layers.l2_regularizer(1e-4)
                                  )(x)
        return x

    def build(self,input_shape) : #,X_input
        
        inputs = keras.layers.Input(input_shape)  
        x = keras.layers.Conv2D(filters=self.in_filter,
                                   kernel_size=3,
                                   padding="SAME", strides=1,
                                   kernel_initializer='he_normal',
                                   kernel_regularizer=tf.contrib.layers.l2_regularizer(1e-4)
                                  )(inputs)
#         x = tf.layers.batch_normalization(inputs = x, training=self.training)
        x = keras.layers.BatchNormalization()(x,training=self.training)
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
        x = keras.layers.BatchNormalization()(x,training=self.training) #
        x = keras.layers.Activation('relu')(x)
        x = keras.layers.AveragePooling2D(pool_size = 8)(x)
        x = keras.layers.Flatten()(x)
        out = keras.layers.Dense(units=1, activation='sigmoid',kernel_initializer ='he_normal')(x) 
        # out = tf.nn.sigmoid(x,name='output-layer')
        
        # keras model summary
        model = keras.Model(inputs=inputs, outputs=out)
        # print(model.summary())
        
        return model


""" ResNet tf.tensorflow """
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