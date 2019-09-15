import keras
import tensorflow as tf
from keras import backend as K

""" Inception v4  """
class InceptionV4():
    def __init__(self,is_train):
        self.is_train = is_train
        self.name = 'InceptionV4'
        self.k = 192
        self.i = 224
        self.m = 256
        self.n = 384
    
    def conv_block(self, x, ksize, fsize, stride, pad="valid", acti=True, batch=False):
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
        x = self.conv_block(x, 3, 32, 2, "valid")
        x = self.conv_block(x, 3, 32, 1, "valid")
        x = self.conv_block(x, 3, 64, 1, "same")
        
        node1 = self.pooling(x, 3, 2, "valid", types="MAX")
        node2 = self.conv_block(x, 3, 64, 2, "valid")
        
        x = tf.concat((node1, node2),axis=-1) # stem_block1 73x73x160
        
        node1 = self.conv_block(x, 1, 64, 1, "same")
        node1 = self.conv_block(node1, (7,1), 64, 1, "same")
        node1 = self.conv_block(node1, (1,7), 64, 1, "same")
        node1 = self.conv_block(node1, 3, 96, 1, "valid")
        
        node2 = self.conv_block(x, 1, 64, 1, "same")
        node2 = self.conv_block(node2, 3, 96, 1, "valid")
        
        x = tf.concat((node1, node2), axis=-1) # stem_block2 71x71x192
        
        node1 = self.pooling(x, 3, 2, "valid", types="MAX")
        node2 = self.conv_block(x, 3, 192, 2, "valid")
        
        x = tf.concat((node1, node2),axis=-1) # stem_block3 71x71x192
        
        return x
    
    def inception_A(self,x):
        node1 = self.conv_block(x,1,96,1,"same")
        node1 = self.pooling(node1,3,1,"same",types="AVG")

        node2 = self.conv_block(x, 1, 96, 1, "same")

        node3 = self.conv_block(x, 3, 96, 1, "same")
        node3 = self.conv_block(node3, 1, 64, 1, "same")

        node4 = self.conv_block(x, 3, 96, 1, "same")
        node4 = self.conv_block(node4, 3, 96, 1, "same")
        node4 = self.conv_block(node4, 1, 64, 1, "same")

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
        node1 = self.pooling(x, 3, 2, "valid",types="MAX")
        
        node2 = self.conv_block(x,3,self.n,2,"valid")
        
        node3 = self.conv_block(x, 3, self.m, 2, "valid")
        node3 = self.conv_block(node3, 3, self.i, 1, "same")
        node3 = self.conv_block(node3, 1, self.k, 1, "same")
        
        return tf.concat((node1, node2, node3),axis=-1) # 71x71x192
        
    def reduction_B(self,x):
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
