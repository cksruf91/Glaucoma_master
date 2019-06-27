import numpy as np
import os
import timeit
import time
from sklearn import metrics
from sklearn.metrics import confusion_matrix
# import argparse

from config import *
from model import *
from iterator import batch_iterator
from utils.util import train_progressbar, slack_message, learning_rate_schedule, print_confusion_matrix


# def args():
#     parser = argparse.ArgumentParser()
#     parser.add_argument('--model', type=int, required=True, choices=[0,1]
#                         , help='number of class')  # number of class
#     args = parser.parse_args()
#     return args
# arg = args()

opts = TrainOption()

import tensorflow as tf
from tensorflow.python.client import device_lib
print(device_lib.list_local_devices())

def auc_function(y_target, y_pred):
    fpr, tpr, thresholds = metrics.roc_curve(y_target, y_pred, pos_label=1)
    return metrics.auc(fpr, tpr) 


with tf.device('/device:GPU:0'):
    # placeholder for images
    shapes = list((None,)+IMAGE_SHAPE)
    images = tf.placeholder('float32', shape=shapes, name='images')  
    
    # placeholder for labels
    labels = tf.placeholder('float32', shape=[None, 1], name='labels')  
    
    # placeholder for training boolean (is training)
    training = tf.placeholder('bool', name='training') 
    
    global_step = tf.get_variable(name='global_step', shape=[], dtype='int64', trainable=False)  
    learning_rate = tf.placeholder('float32', name='images')
    # learning_rate = tf.train.exponential_decay(opts.LEARNING_RATE, global_step, opts.LR_DEACY_STEPS, opts.LR_DECAY_RATE)
    
    ## placeholder fot store Beat accuracy
    best_score = tf.get_variable(name='best_accuracy', dtype='float32', trainable=False, initializer=0.0)
    
    # model build
    with tf.variable_scope('build'):
#         output = ResNetV3(training).build(images)
        output = InceptionV4(training).build(images)

    

#loss and optimizer
with tf.variable_scope('losses'):
#     loss = tf.keras.backend.binary_crossentropy(labels, output)
    loss = tf.nn.weighted_cross_entropy_with_logits(labels, output,pos_weight=15.0 )
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
    
#accuracy
# with tf.variable_scope('accuracy'):
#     output = tf.nn.softmax(output, name='output')
#     prediction = tf.equal(tf.argmax(output, 1), tf.argmax(labels, 1), name='prediction')
#     accuracy = tf.reduce_mean(tf.cast(prediction, tf.float32), name='accuracy')
#     auc = tf.metrics.auc(labels, output, name='auc')
    
# method to save model
# 참조 : https://goodtogreate.tistory.com/entry/Saving-and-Restoring
saver = tf.train.Saver()


## train 데이터 사이즈를 따로 저장하여 불러옴
with open(TRAIN_FILE.split('.')[0] + '.length', 'r') as f:
    train_data_lenth = int(f.read())
    
with open(TEST_FILE.split('.')[0] + '.length', 'r') as f:
    test_data_lenth = int(f.read())


with tf.Session() as sess:
    sess.run(tf.global_variables_initializer())
    gstep = sess.run(global_step)
    
    ## load batch generator
    print(f"train data from : {TRAIN_FILE}")
    train_iterator = batch_iterator(TRAIN_FILE 
                                    , batch_size=opts.BATCH_SIZE
                                    , training=True, shuffle=True, buffer_size=50)
    train_images_batch, train_labels_batch = train_iterator.get_next()
    
    print(f"test data from : {TEST_FILE}")
    test_iterator = batch_iterator(TEST_FILE 
                                   , batch_size=opts.BATCH_SIZE
                                   , training=False, shuffle=False, buffer_size=50)
    test_images_batch, test_labels_batch = test_iterator.get_next()
    
    ## initialize batch generator
    sess.run(train_iterator.initializer)
    sess.run(test_iterator.initializer)
    
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
            step_per_epoch = train_data_lenth//opts.BATCH_SIZE
        else:
            step_per_epoch = opts.STEP_PER_EPOCH
            
        ## learning late schedule
        lr = learning_rate_schedule(epoch_, opts.LEARNING_RATE)
        print(f' Current Learning Rate : {lr}')
        
        ## epoch당 step 계산
        step = 1
        train_loss = []
        train_acc = np.array([])
        
        y_pred = np.array([])
        y_target = np.array([])
        
        start = timeit.default_timer()
        
        """ Train """
        while step < step_per_epoch:
            train_images, train_labels = sess.run([train_images_batch, train_labels_batch])
            # accuracy_
            gstep, _, loss_, output_ = sess.run(
                        [global_step, train_op, loss, output],
                        feed_dict={images: train_images, labels: train_labels
                                   , learning_rate : lr, training: True})
                            #, tf.keras.backend.learning_phase():1
            
            y_pred = np.append(y_pred , np.array(output_).flatten())
            y_target = np.append(y_target , np.array(train_labels).flatten())
            
            ## losses
            train_loss.append(loss_)
            mean_train_loss = np.mean(train_loss)
            
            ## accuracy
            equal = np.equal(y_pred>0.5, y_target)
            train_acc =np.append(train_acc, equal)
            mean_train_acc = np.mean(train_acc)
            
            ## EPOCH 진행상황 출력
            train_progressbar(step, step_per_epoch
                              , epoch_, EPOCHS
                              ,mean_train_loss ,mean_train_acc , 1, 25)
            step += 1
        
        ## auc 
        auc = auc_function(y_target, y_pred)
        epoch_time = time.strftime("%H:%M:%S", time.gmtime(timeit.default_timer()-start))
        
        
        val_loss = []
        val_acc = np.array([])
        
        y_pred = np.array([])
        y_target = np.array([])
        
        """ Validation """
        while True:
            try:
                test_images, test_labels = sess.run([test_images_batch, test_labels_batch])
                loss_, output_= sess.run([loss, output]
                                         , feed_dict={images: test_images, labels: test_labels
                                       , training: False}) #tf.keras.backend.learning_phase():0

                ## losses
                val_loss.append(loss_)
                
                y_pred = np.append(y_pred , np.array(output_).flatten())
                y_target = np.append(y_target , np.array(test_labels).flatten())
                
                ## accuracy
                #equal = np.equal(np.argmax(output_,axis =1) , np.argmax(train_labels,axis =1))
                equal = np.equal(y_pred>0.5, y_target)
                val_acc =np.append(val_acc, equal)

                
            except tf.errors.OutOfRangeError:
                sess.run(test_iterator.initializer)
                break
        
        mean_val_loss = np.mean(val_loss)
        mean_val_acc = np.mean(val_acc)
        print(f"\t- val loss : {mean_val_loss:5.5f}\t- val acc : {mean_val_acc:5.5f}")
        
        ## auc 
        val_auc = auc_function(y_target, y_pred)
        
        print('time: ',epoch_time, f"\ttrain auc : {auc:5.5f},\tvalidation auc : {val_auc:5.5f}"
              ,'\tbest :', best_score.eval(),  )
        
        # confusion_matrix(y_treu , y_pred)
        print_confusion_matrix(y_target,y_pred)
        
        
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
        
        if epoch_ % 5 ==0 :
            time.sleep(60*2)
            
print('end of line..')
