import os
import sys
sys.path = ["C:\\Users\\infinigru\\Anaconda3\\envs\\prac\\lib\\site-packages"] + sys.path
sys.path.append("..")
import numpy as np
import cv2
import random
# from skimage.transform import rescale, resize, downscale_local_mean
import tensorflow as tf

from config import *
from utils.util import progress

#TODO
# Local Histogram Equalization
# CLAHE (Contrast Limited Adaptive Histogram Equalization)
def Adaptive_Histogram_Equalization(img):
    clahe = cv2.createCLAHE(clipLimit=1, tileGridSize=(8,8))
    for i in range(3):
        ch = img[:,:,i]
        ch = clahe.apply(ch)
        img[:,:,i] = ch
    return img 

def resize_img(img,shape):
    # cv2.INTER_NEAREST -- 이웃 보간법
    # cv2.INTER_LINEAR -- 쌍 선형 보간법
    # cv2.INTER_LINEAR_EXACT -- 비트 쌍 선형 보간법
    # cv2.INTER_CUBIC -- 바이큐빅 보간법
    # cv2.INTER_AREA -- 영역 보간법
    # cv2.INTER_LANCZOS4 -- Lanczos 보간법
    # 기본적으로 쌍 선형 보간법이 가장 많이 사용됩니다.
    # 이미지를 확대하는 경우, 바이큐빅 보간법이나 쌍 선형 보간법을 가장 많이 사용합니다.
    # 이미지를 축소하는 경우, 영역 보간법을 가장 많이 사용합니다.
    # 영역 보간법에서 이미지를 확대하는 경우, 이웃 보간법과 비슷한 결과를 반환합니다.
    # 출처 : https://076923.github.io/posts/Python-opencv-8/
    img = cv2.resize(img, dsize=shape, interpolation=cv2.INTER_AREA)
    #img = resize(img,shape)
    return img

def log_transform(img):
    img = np.log(img+1)
    return img
    
def normalize_img(img):
#     image = np.log10(image+1)
#     shape = img.shape
#     img = np.float64(img.reshape(-1))
#     img -= img.mean()
#     img /= img.std()
#     img = img.reshape(shape)
#     img = tf.image.per_image_standardization(img)
    img = img/ 255
    return img

def c_normalize(img):
    img = img.astype(float)
    for i in range(3):
        ch = img[:,:,i] - img[:,:,i].mean()
        ch = ch /img[:,:,i].std()
        img[:,:,i] = ch
    return img

def int64_feature(value):
    return tf.train.Feature(int64_list=tf.train.Int64List(value=[value]))

def int64_list_feature(value):
    return tf.train.Feature(int64_list=tf.train.Int64List(value=value))

def bytes_feature(value):
    return tf.train.Feature(bytes_list=tf.train.BytesList(value=[value]))

def bytes_list_feature(value):
    return tf.train.Feature(bytes_list=tf.train.BytesList(value=value))

def float_list_feature(value):
    return tf.train.Feature(float_list=tf.train.FloatList(value=value))

### segmentaion ###
def segment_tfrecord(image_dir, masking_dir, output_file):
    writer = tf.python_io.TFRecordWriter(output_file)
    
    image_files = []
    mask_files = []
    for mkfile in os.listdir(masking_dir):
        imfile = mkfile.split('_')[-1]
        mask_files.append(os.path.join(masking_dir,mkfile))
    
        for (path, dir, files) in os.walk(image_dir):
            if path ==image_dir: # 현재 디렉토리는 넘김
                continue
            for file in files:
                if file == imfile:
                    image_files.append(os.path.join(path, file))
    
    pr = progress()
    total = len(image_files)
    count = 0
    for image, mask in zip(image_files, mask_files):
        count += 1
        pr.print_progress(1,total,count)
        
        image = cv2.imread(image, cv2.IMREAD_COLOR)
        mask = cv2.imread(mask, cv2.IMREAD_COLOR)
        
        ## image 사이즈를 균일하게 맞춤
        image = resize_img(image,IMAGE_SHAPE[:2])
        
        mask = resize_img(mask,IMAGE_SHAPE[:2])
        mask = mask[:,:,0]
        mask = np.where(mask>0,1,mask)
        mask = mask[:,:,np.newaxis]    
        
        ## Adaptive_Histogram_Equalization
        image = Adaptive_Histogram_Equalization(image)
        
        ## image 정규화
        image = normalize_img(image)
        
        
        height = image.shape[0]
        width = image.shape[1]
        
        example = tf.train.Example(features=tf.train.Features(feature={
                        'height': int64_feature(height),
                        'width': int64_feature(width),
                        'image/raw': bytes_feature(image.tobytes()),
                        'mask': bytes_feature(mask.tobytes())
                        }))
        writer.write(example.SerializeToString())
        
    print('finish')
    writer.close()
    
    ## step_per_epoch 계산을위해 데이터의 length를 따로 저장 
    out_dirctory = os.path.dirname(output_file)
    out_filename = os.path.basename(output_file).split('.')[0] + '.length'

    with open(os.path.join(out_dirctory,out_filename) , 'w') as f:
        f.write(str(count))
    
def segment_parse_tfrecord(record):

    keys_to_features = {
        'height': tf.FixedLenFeature((), tf.int64),
        'width': tf.FixedLenFeature((), tf.int64),
        'image/raw': tf.FixedLenFeature((), tf.string),
        'mask': tf.FixedLenFeature((), tf.string),
        # 'label': tf.FixedLenFeature((), tf.string),
    }

    features = tf.parse_single_example(record, features=keys_to_features)
    height = tf.cast(features['height'], tf.int64)
    width = tf.cast(features['width'], tf.int64)
    image = tf.cast(features['image/raw'], tf.string)
    mask = tf.cast(features['mask'], tf.string)
    
    image = tf.decode_raw(image, tf.float64)
    image = tf.reshape(image, shape=[height, width, -1])
    
    mask = tf.decode_raw(mask, tf.uint8)
    mask = tf.reshape(mask, shape=[height, width, -1])

    return image, mask

if __name__ == "__main__":
    
    print('generate segment tfrecord file...')
    segment_tfrecord(TRAIN_IMAGE, MASKING_TRAIN_IMAGE, SEGMENT_TRAIN_FILE)
    segment_tfrecord(TRAIN_IMAGE, MASKING_VAL_IMAGE, SEGMENT_TEST_FILE)