import numpy as np
import os
import timeit
import time
from sklearn import metrics
from sklearn.metrics import confusion_matrix
# import argparse

from config import *
from model import *
# from iterator import batch_iterator
from iterator import DataGenerator
# from utils.tfrecord_util import parse_tfrecord
from utils.util import train_progressbar, slack_message, learning_rate_schedule, print_confusion_matrix

# def args():
#     parser = argparse.ArgumentParser()
#     parser.add_argument('--model', type=int, required=True, choices=[0,1]
#                         , help='number of class')  # number of class
#     args = parser.parse_args()
#     return args
# arg = args()

opts = TrainOption('cls')

import tensorflow as tf
from tensorflow.python.client import device_lib
print(device_lib.list_local_devices())

def auc_function(y_target, y_pred):
    fpr, tpr, thresholds = metrics.roc_curve(y_target, y_pred, pos_label=1)
    return metrics.auc(fpr, tpr) 


with tf.device('/device:GPU:0'):
    # placeholder for images
    shapes = list((None,)+OPTIC_DISC_SHAPE)
    images = tf.placeholder('float32', shape=shapes, name='images')  
    
    # placeholder for labels
    labels = tf.placeholder('float32', shape=[None, 1], name='labels')  
    
    # placeholder for training boolean (is training)
    training = tf.placeholder('bool', name='training') 
    
    global_step = tf.get_variable(name='global_step', shape=[], dtype='int64', trainable=False)  
    learning_rate = tf.placeholder('float32', name='learning_rate')
    # learning_rate = tf.train.exponential_decay(opts.LEARNING_RATE, global_step, opts.LR_DEACY_STEPS, opts.LR_DECAY_RATE)
    
    ## placeholder fot store Beat accuracy
    best_score = tf.get_variable(name='best_accuracy', dtype='float32', trainable=False, initializer=0.0)
    
    # model build
    with tf.variable_scope('build'):
        output = ResNetV2(training).build(images)
#         output = InceptionV4(training).build(images)
    
    print('model build')
    
#loss and optimizer
with tf.variable_scope('losses'):
    loss = tf.keras.backend.binary_crossentropy(labels, output)
#     loss = tf.nn.weighted_cross_entropy_with_logits(labels = labels, logits = output, pos_weight=100.0 )
    loss = tf.reduce_mean(loss, name='loss')
#     loss = tf.nn.softmax_cross_entropy_with_logits_v2(labels=labels
#                                                       , logits=output
#                                                       , name='cross_entropy')
#     loss = tf.losses.softmax_cross_entropy(labels, output, label_smoothing=0.1)
#     l2_loss = tf.add_n([tf.nn.l2_loss(var) for var in tf.trainable_variables()], name='l2_loss')
#     loss = loss + l2_loss * opts.WEIGHT_DECAY

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
    
    ## load batch generator
    print(f"train data from : {TRAIN_IMAGE}")
    train_iterator = DataGenerator(TRAIN_IMAGE, MASK_LOC ,opts.BATCH_SIZE
                                   ,IMAGE_SHAPE, OPTIC_DISC_SHAPE, is_train=True, copy = True, sample=20)
    
    print(f"test data from : {TEST_IMAGE}")
    test_iterator = DataGenerator(TEST_IMAGE, MASK_LOC ,opts.BATCH_SIZE 
                                   ,IMAGE_SHAPE, OPTIC_DISC_SHAPE, is_train=False, copy = False, sample=20)
    
    ## try to restore last model checkpoint
    try: 
        saver.restore(sess, tf.train.latest_checkpoint(RESULT_PATH))
        check_point_name = tf.train.latest_checkpoint(RESULT_PATH)
        last_epoch = int(check_point_name.split('_')[-1].split('.')[0])
        print("checkpoint restored")
    except:
        last_epoch = 0        
        print("failed to load checkpoint")
    
    ## epoch
    EPOCHS = opts.EPOCHS

    """ run train """
    for epoch_ in range(EPOCHS - last_epoch):
        
        epoch_ += 1+last_epoch

        if opts.STEP_PER_EPOCH is None:
            step_per_epoch = len(train_iterator)
        else:
            step_per_epoch = opts.STEP_PER_EPOCH
            
        ## learning late schedule
        lr = learning_rate_schedule(epoch_, opts.LEARNING_RATE)
        print(f' Current Learning Rate : {lr}')
        
        ## epoch당 step 계산
        step = 1
        train_loss = []
        train_acc = np.array([])
        
        train_y_pred = np.array([])
        train_y_target = np.array([])
        
        start = timeit.default_timer()
        
        """ Train """
        while step <= step_per_epoch:
            
            train_images, train_labels = train_iterator[step]
            gstep, _, loss_, output_ = sess.run( [global_step, train_op, loss, output],
                                                feed_dict={images: train_images, labels: train_labels
                                                           , learning_rate : lr, training: True})
                            #, tf.keras.backend.learning_phase():1
            
            train_y_pred = np.append(train_y_pred , np.array(output_).flatten())
            train_y_target = np.append(train_y_target , np.array(train_labels).flatten())
            
            ## losses
            train_loss.append(loss_)
            mean_train_loss = np.mean(train_loss)
            
            ## accuracy
            equal = np.equal(train_y_pred>0.5, train_y_target)
            train_acc =np.append(train_acc, equal)
            mean_train_acc = np.mean(train_acc)
            
            ## EPOCH 진행상황 출력
            train_progressbar(step, step_per_epoch
                              , epoch_, EPOCHS
                              ,mean_train_loss, 'acc' ,mean_train_acc , 1, 25)
            step += 1
            
        train_iterator.on_epoch_end()
        
        ## auc 
        auc = auc_function(train_y_target, train_y_pred)
        epoch_time = time.strftime("%H:%M:%S", time.gmtime(timeit.default_timer()-start))        
        
        val_loss = []
        val_acc = np.array([])
        
        val_y_pred = np.array([])
        val_y_target = np.array([])
        
        """ Validation """
        while True:
            try:
                test_images, test_labels = test_iterator[step]
                loss_, output_= sess.run([loss, output]
                                         , feed_dict={images: test_images, labels: test_labels
                                       , training: False}) #tf.keras.backend.learning_phase():0

                ## losses
                val_loss.append(loss_)
                
                val_y_pred = np.append(val_y_pred , np.array(output_).flatten())
                val_y_target = np.append(val_y_target , np.array(test_labels).flatten())
                
                ## accuracy
                #equal = np.equal(np.argmax(output_,axis =1) , np.argmax(train_labels,axis =1))
                equal = np.equal(val_y_pred>0.5, val_y_target)
                val_acc =np.append(val_acc, equal)

            except IndexError:
                test_iterator.on_epoch_end()
                break
        
        mean_val_loss = np.mean(val_loss)
        mean_val_acc = np.mean(val_acc)
        print(f"\t- val loss : {mean_val_loss:5.5f}\t- val acc : {mean_val_acc:5.5f}")
        
        ## auc 
        val_auc = auc_function(val_y_target, val_y_pred)
        
        print('time : ',epoch_time,"||", f"\ttrain auc : {auc:1.5f},\tvalidation auc : {val_auc:1.5f}"
              ,'\tbest :', best_score.eval() )
        
        # confusion_matrix(y_treu , y_pred)
        print('train confustion_matrix')
        print_confusion_matrix(train_y_target,train_y_pred,0.5)
        print('validation confustion_matrix')
        print_confusion_matrix(val_y_target,val_y_pred,0.5)
        
        
        #### end of epoch functions ####
        ## early stop training
        patience = 0
        if best_score.eval() < val_auc:
            best_score = tf.assign(best_score, val_auc)
        else:
            patience += 1
    
        ## send result to slack
        if SEND_MESSAGE:
            message = "epoch : {} | time : {} \n matrix : {:5.5f}".format(epoch_, epoch_time, confusion_matrix(y_pred>0.5,y_target))
            slack_message('#resnet_project', message)
        
        # save checkpoint
        if SAVE_CHECKPOINT:
            save_path = saver.save(sess, os.path.join(RESULT_PATH,f'checkpoint_{epoch_}.ckpt'))
        
        # save history 
        history_line = str(epoch_)+','+str(mean_train_loss)+','+str(auc)+','+ str(mean_val_loss)+','+str(val_auc)+'\n'
        if epoch_==1:
            history_line = "epoch,loss,auc,val_loss,val_auc\n" + history_line
        with open(os.path.join(RESULT_PATH,"history.csv"),'a') as f:
            f.write(history_line)      
        
        if EARLY_STOPPING:
            if patience > 7:
                break
        
#         if epoch_ % 5 ==0 :
#             time.sleep(60*2)
            
print('end of line..')
