import numpy as np
import os 
from scipy.ndimage import rotate, shift
import cv2
import random

# import keras
import tensorflow as tf

from config import *
from utils.tfrecord_util import parse_tfrecord
from utils.util import progress

def pad_and_crop(image, shape, pad_size=2):

    image = tf.image.pad_to_bounding_box(image, pad_size, pad_size, shape[0]+pad_size*2, shape[1]+pad_size*2)
    image = tf.image.random_crop(image, shape)
    return image

def random_rotate(image,angle):
#     rotate_angle = random.choice(range(0,180,10))
#     image = rotate(image ,rotate_angle ,reshape=False ,mode='reflect')
    rotate_angle = tf.math.round(tf.random.uniform([],0,angle))
    image = tf.contrib.image.rotate(image,
                            rotate_angle,
                            interpolation='NEAREST')
    return image


def random_flip(image):    
    image = tf.image.random_flip_left_right(image) #좌우 반전
    image = tf.image.random_flip_up_down(image) #상하 반전
    
    # random_value = random.random()
    # image = cv2.flip(image, 0) 
    # image = cv2.flip(image, 1) 
    return image

def random_image_shift(image):
    tx = tf.random.shuffle([-3,-2,-1,0,1,2,3])
    ty = tf.random.shuffle([-3,-2,-1,0,1,2,3])
    transforms = [1, 0, tx[0], 0, 1, ty[0], 0, 0]
    image = tf.contrib.image.transform(image, transforms, interpolation='NEAREST')
#     image = shift(image,shift =[tx,ty,0])
    return image


def image_preprocess(image,label):  
    
        ## random한 각도로 image를 회전
    image = random_rotate(image,350)
    
        ## image shift
    image = random_image_shift(image)
    
        ## 50% 확률로 이미지 상하 혹은 좌우 반전
    image = random_flip(image)

    ## 이미지 자르기
    # image = pad_and_crop(image,IMAGE_SHAPE,pad_size=2)
    
    return image, label

def batch_iterator(infile , batch_size, training, shuffle,buffer_size):
    if os.path.isfile(infile) is False:
        raise FileNotFoundError(infile, 'not exist')
    
    dataset = tf.data.TFRecordDataset(infile)
    dataset = dataset.map(parse_tfrecord)        
    if training:
        dataset = dataset.map(image_preprocess, num_parallel_calls = -1) #tf.data.experimental.AUTOTUNE
        dataset = dataset.repeat()
        if shuffle:
            dataset = dataset.shuffle(buffer_size)
    dataset = dataset.batch(batch_size, drop_remainder=False)
    iterator = dataset.make_initializable_iterator()
    return iterator



