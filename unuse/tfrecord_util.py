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

def crop_optic_disk(img,mk):
    h = np.where(mk>0)[0]
    h = int(mk.shape[0]/2) if h.size == 0 else h
        
    w = np.where(mk>0)[1]
    w = int(mk.shape[1]/2) if w.size == 0 else w
    
    margin = 3
    maxh = min(np.max(h)+margin, mk.shape[0])
    minh = max(np.min(h)-margin, 0)
    maxw = min(np.max(w)+margin, mk.shape[1])
    minw = max(np.min(w)-margin, 0)
    
    img = img[minh:maxh,minw:maxw,:]
    return resize_img(img, OPTIC_DISC_SHAPE[:2])

def log_transform(img):
    img = np.log(img+1)
    return img
    
def normalize_img(img):
#     shape = img.shape
#     img = np.float64(img.reshape(-1))
#     img -= img.mean()
#     img /= img.std()
#     img = img.reshape(shape)
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

def parse_tfrecord(record):

    keys_to_features = {
        'height': tf.FixedLenFeature((), tf.int64),
        'width': tf.FixedLenFeature((), tf.int64),
        'image/raw': tf.FixedLenFeature((), tf.string),
        'image/mask': tf.FixedLenFeature((), tf.string),
        'label': tf.FixedLenFeature((), tf.string),
    }

    features = tf.parse_single_example(record, features=keys_to_features)
    height = tf.cast(features['height'], tf.int64)
    width = tf.cast(features['width'], tf.int64)
    image = tf.cast(features['image/raw'], tf.string)
    mask = tf.cast(features['image/mask'], tf.string)
    label = tf.cast(features['label'], tf.string)

    image = tf.decode_raw(image, tf.float64)
    image = tf.reshape(image, shape=[height, width, -1])
    
    mask = tf.decode_raw(mask, tf.uint8)
    mask = tf.reshape(mask, shape=[height, width, -1])
    
    label = tf.decode_raw(label, tf.int32)
    label = tf.reshape(label, shape=[1,])

    return image, mask, label

#     def image_and_label(image_files):
#         ## get lebel from file name 
#         images_label = []
#         for i,names in enumerate(image_files):
#                 #images_label.append(names.split("_")[3].split(".")[0])
#             images_label.append(os.path.dirname(names).split('\\')[-1])

#         for image_name, label_list in zip(image_files, images_label):
#             image = cv2.imread(image_name, cv2.IMREAD_COLOR)
#             label = LABEL[label_list]
#             yield image, label

def create_tfrecord(image_dir, mask_dir, output_file, copy = False):
    
    writer = tf.python_io.TFRecordWriter(output_file)
    #image_files = os.listdir(image_dir)
    
    image_files = []

    for (path, dir, files) in os.walk(image_dir):
        if path ==image_dir: # 현재 디렉토리는 넘김
            continue
        numcopy = 9 if (copy and path.split('\\')[-1] == "Glaucoma") else 1
        for file in files*numcopy:
            image_files.append(os.path.join(path, file))
            
    random.shuffle(image_files)

    pr = progress()
    total = len(image_files)
    count = 0
    for file_name in image_files:
        
        ## get image
        image = cv2.imread(file_name, cv2.IMREAD_COLOR)
        
        ## get label
        label = os.path.dirname(file_name).split('\\')[-1]
        label = LABEL[label]
        
        ## get mask
        mask = os.path.join(mask_dir, 'mask_'+os.path.basename(file_name))
        mask = cv2.imread(mask, cv2.IMREAD_COLOR)
        
        count += 1
        pr.print_progress(1,total,count)
        
        ## image 사이즈를 균일하게 맞춤
        image = resize_img(image,IMAGE_SHAPE[:2])
        mask = resize_img(mask,IMAGE_SHAPE[:2])
#         print(mask.dtype)
        
#         ## get optic disc
#         image = crop_optic_disk(image,mask)
        
        ## Adaptive_Histogram_Equalization
        #image = Adaptive_Histogram_Equalization(image)
        ## image 정규화
#         image = normalize_img(image)
        image = image.astype('float64')
#         print(mask.max())
        height = image.shape[0]
        width = image.shape[1]
        example = tf.train.Example(features=tf.train.Features(feature={
                        'height': int64_feature(height),
                        'width': int64_feature(width),
                        'image/raw': bytes_feature(image.tobytes()),
                        'image/mask': bytes_feature(mask.tobytes()),
                        'label': bytes_feature(np.array(label).tobytes())
                        #'label': int64_feature(label)
                    }))
        writer.write(example.SerializeToString())
        
    print('\nfinish')
    writer.close()
    
    ## step_per_epoch 계산을위해 데이터의 length를 따로 저장 
    out_dirctory = os.path.dirname(output_file)
    out_filename = os.path.basename(output_file).split('.')[0] + '.length'

    with open(os.path.join(out_dirctory,out_filename) , 'w') as f:
        f.write(str(count))


if __name__ == "__main__":

    print('generate tfrecord file...')    
    create_tfrecord(TRAIN_IMAGE, MASK_LOC, TRAIN_FILE,copy = False)
#     create_tfrecord(TEST_IMAGE, MASK_LOC, TEST_FILE)
