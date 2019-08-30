import numpy as np
import os
import timeit
import time
import cv2
import argparse
from sklearn import metrics
from sklearn.metrics import confusion_matrix

from config import *
from model import *
from seg_iterator import seg_batch_iterator
from utils.seg_tfrecord_util import segment_parse_tfrecord, normalize_img, resize_img
from utils.util import train_progressbar, slack_message, learning_rate_schedule, print_confusion_matrix

def args():
    parser = argparse.ArgumentParser()
    parser.add_argument('-m','--mode', type=int, required=True, choices=[0,1]
                        , help='1: train image, 0: test image')  # number of class
    args = parser.parse_args()
    return args
arg = args()

if arg.mode:
    locs = TRAIN_IMAGE
else :
    locs = TEST_IMAGE

def auc_function(y_target, y_pred):
    fpr, tpr, thresholds = metrics.roc_curve(y_target, y_pred, pos_label=1)
    return metrics.auc(fpr, tpr) 

with tf.device('/device:GPU:0'):
    # placeholder for images
    shapes = list((None,)+IMAGE_SHAPE)
    images = tf.placeholder('float32', shape=shapes, name='images')  
    
    # placeholder for labels
    shapes = list((None,)+IMAGE_SHAPE[:2]+(1,))
    mask = tf.placeholder('float32', shape=shapes, name='mask')
    
    # placeholder for training boolean (is training)
    training = tf.placeholder('bool', name='training') 
    
    global_step = tf.get_variable(name='global_step', shape=[], dtype='int64', trainable=False)  
    learning_rate = tf.placeholder('float32', name='learning_rate')
    # learning_rate = tf.train.exponential_decay(opts.LEARNING_RATE, global_step, opts.LR_DEACY_STEPS, opts.LR_DECAY_RATE)
    
    ## placeholder fot store Beat accuracy
    best_score = tf.get_variable(name='best_accuracy', dtype='float32', trainable=False, initializer=0.0)
    
    # model build
    with tf.variable_scope('build'):
        # output = ResNetV3(training).build(images)
        # output = InceptionV4(training).build(images)
        output = Unet(training).build(images)
    
    print('model build')

    

#loss and optimizer
with tf.variable_scope('losses'):
    loss = tf.keras.backend.binary_crossentropy(mask, output)
    loss = tf.reduce_mean(loss, name='loss')

with tf.variable_scope('optimizers'):
    optimizer = tf.train.AdamOptimizer(learning_rate)
    update_ops = tf.get_collection(tf.GraphKeys.UPDATE_OPS)
    train_op = optimizer.minimize(loss, global_step=global_step) 
    train_op = tf.group([train_op, update_ops], name='train_op')
#     optimizer = tf.train.MomentumOptimizer(learning_rate=opts.LEARNING_RATE, momentum=opts.MOMENTUM, use_nesterov=True)
    
# method to save model
# 참조 : https://goodtogreate.tistory.com/entry/Saving-and-Restoring
saver = tf.train.Saver()


with tf.Session() as sess:
    sess.run(tf.global_variables_initializer())
    gstep = sess.run(global_step)
    
    image_files = []
    for (path, dir, files) in os.walk(locs):
        if path ==TRAIN_IMAGE: # 현재 디렉토리는 넘김
            continue
        for file in files:
            image_files.append(os.path.join(path, file))
    
    ## try to restore last model checkpoint
    try: 
        saver.restore(sess, tf.train.latest_checkpoint(SEGMENT_RESULT_PATH))
        check_point_name = tf.train.latest_checkpoint(SEGMENT_RESULT_PATH)
        last_epoch = int(check_point_name.split('_')[-1].split('.')[0])
        print("checkpoint restored")
    except:
        last_epoch = 0        
        print("failed to load checkpoint")

    
    for file in image_files:
        image = cv2.imread(file, cv2.IMREAD_COLOR)
        image = resize_img(image,IMAGE_SHAPE[:2])
        image = normalize_img(image)
        
        image = image[np.newaxis,:,:,:]
        
        """ Validation """
        output_= sess.run([output], feed_dict={images: image, training: False}) 
        
        masking = output_[0]
        m = masking[0]
        m2 = np.append(m,m,axis =2)
        m2 = np.append(m2,m,axis =2)
        m2 = np.where(m2>0.5,255,m2)
        
        masking = m2
        
        
        file = os.path.basename(file)
        print(cv2.imwrite(os.path.join(MASK_LOC ,'mask_'+ file),masking))

            
print('end..')
