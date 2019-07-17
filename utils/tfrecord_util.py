from __future__ import print_function

import random
import sys

import cv2
import numpy as np

sys.path = ['C:\\Users\\infinigru\\Anaconda3\\envs\\prac\\lib\\site-packages'] + sys.path
sys.path.append('..')

# from skimage.transform import rescale, resize, downscale_local_mean

import tensorflow as tf

from config import *
from utils.util import Progress


def resize_img(img, shape):
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
    #     img = resize(img,shape)
    return img


def log_transform(img):
    img = np.log(img + 1)
    return img


def normalize_img(img):
    #     image = np.log10(image+1)
    shape = img.shape
    img = np.float64(img.reshape(-1))
    img -= img.mean()
    img /= img.std()
    img = img.reshape(shape)
    #     img = tf.image.per_image_standardization(img)
    #     img = img/ 255
    return img


def c_normalize(img):
    img = img.astype(float)
    for i in range(3):
        ch = img[:, :, i] - img[:, :, i].mean()
        ch = ch / img[:, :, i].std()
        img[:, :, i] = ch
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
        'label': tf.FixedLenFeature((), tf.string),
    }

    features = tf.parse_single_example(record, features=keys_to_features)
    height = tf.cast(features['height'], tf.int64)
    width = tf.cast(features['width'], tf.int64)
    image = tf.cast(features['image/raw'], tf.string)
    label = tf.cast(features['label'], tf.string)

    image = tf.decode_raw(image, tf.float64)
    image = tf.reshape(image, shape=[height, width, -1])

    label = tf.decode_raw(label, tf.int32)
    label = tf.reshape(label, shape=[10, ])

    return image, label


def create_tfrecord(image_dir, output_file):
    def image_and_label(image_filenames):
        ## get label from file name
        images_label = [os.path.dirname(names).split('\\')[-1]
                        # images_label.append(names.split('_')[3].split('.')[0])
                        for i, names in enumerate(image_filenames)
                        ]

        for image_name, label_list in zip(image_filenames, images_label):
            image_obj = cv2.imread(image_name, cv2.IMREAD_COLOR)
            label = LABEL[label_list]
            yield image_obj, label

    writer = tf.python_io.TFRecordWriter(output_file)

    #     image_filenames = os.listdir(image_dir)
    image_files = []
    for (path, _directory, files) in os.walk(image_dir):
        if path == image_dir:
            continue
        image_files += [os.path.join(path, file) for file in files]
    random.shuffle(image_files)

    pr = Progress()
    total = len(image_files)
    count = 0
    for image, label in image_and_label(image_files):
        ## image 사이즈를 균일하게 맞춤
        image = resize_img(image, IMAGE_SHAPE[:2])
        ## image log 변환
        image = log_transform(image)
        ## image 정규화
        image = normalize_img(image)

        height = image.shape[0]
        width = image.shape[1]
        example = tf.train.Example(features=tf.train.Features(feature={
            'height': int64_feature(height),
            'width': int64_feature(width),
            'image/raw': bytes_feature(image.tobytes()),
            'label': bytes_feature(np.array(label).tobytes())
        }))
        writer.write(example.SerializeToString())
        count += 1
        pr.print_progress(1, total, count)
        # print('\r{0} done'.format(count), end='')
    print('\nfinish')
    writer.close()

    ## step_per_epoch 계산을위해 데이터의 length를 따로 저장 
    out_directory = os.path.dirname(output_file)
    out_filename = '{}.length'.format(out_directory.split('.')[0])
    # print(os.path.join(out_directory,out_filename))
    with open(os.path.join(out_directory, out_filename), 'w') as f:
        f.write(str(count))


if __name__ == '__main__':
    #     if os.path.isfile(TRAIN_FILE):
    #         os.popen(f'del {TRAIN_FILE}')

    #     if os.path.isfile(TEST_FILE):
    #         os.popen(f'del {TEST_FILE}')
    print('generate tfrecord file...')

    create_tfrecord(TRAIN_IMAGE, TRAIN_FILE)
    create_tfrecord(TEST_IMAGE, TEST_FILE)
