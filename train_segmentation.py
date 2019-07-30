import numpy as np
import os
import timeit
import time
from sklearn import metrics
from sklearn.metrics import mean_squared_error
from sklearn.metrics import confusion_matrix

from config import *
from model import *
from seg_iterator import DataGenerator
# from utils.seg_tfrecord_util import segment_parse_tfrecord
from utils.util import train_progressbar, slack_message, learning_rate_schedule, print_confusion_matrix

opts = TrainOption('seg')

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
    best_score = tf.get_variable(name='best_accuracy', dtype='float32', trainable=False, initializer=1.0)
    
    # model build
    with tf.variable_scope('build'):
        # output = ResNetV3(training).build(images)
        # output = InceptionV4(training).build(images)
        output = Unet(training).build(images)
    
    print('model build')

    

#loss and optimizer
with tf.variable_scope('losses'):
    loss = tf.keras.backend.binary_crossentropy(mask, output)
#     loss = tf.nn.weighted_cross_entropy_with_logits(labels, output,pos_weight=15.0 )
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
patience = 0

with tf.Session() as sess:
    sess.run(tf.global_variables_initializer())
    gstep = sess.run(global_step)
    
    ## load batch generator
    print(f"train data from : {MASKING_TRAIN_IMAGE}")
    train_iterator = DataGenerator(TRAIN_IMAGE, MASKING_TRAIN_IMAGE
                                   ,opts.BATCH_SIZE
                                   ,IMAGE_SHAPE, is_train=True,sample = None)
    
    print(f"test data from : {MASKING_VAL_IMAGE}")
    test_iterator = DataGenerator(TRAIN_IMAGE, MASKING_VAL_IMAGE
                                   ,opts.BATCH_SIZE
                                   ,IMAGE_SHAPE, is_train=True,sample = None)
    
    ## try to restore last model checkpoint
    try: 
        saver.restore(sess, tf.train.latest_checkpoint(SEGMENT_RESULT_PATH))
        check_point_name = tf.train.latest_checkpoint(SEGMENT_RESULT_PATH)
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
        
        mse = []
        start = timeit.default_timer()
        
        """ Train """
        while step <= step_per_epoch:
            train_images, train_labels = train_iterator.get_item()
            # accuracy_
            gstep, _, loss_, output_ = sess.run(
                        [global_step, train_op, loss, output],
                        feed_dict={images: train_images, mask: train_labels
                                   , learning_rate : lr, training: True})
                            #, tf.keras.backend.learning_phase():1
            
            y_pred = np.array(output_).flatten()
            y_target = np.array(train_labels).flatten()
            
            ## losses
            train_loss.append(loss_)
            mean_train_loss = np.mean(train_loss)
            
            ## mse
            mse.append(mean_squared_error(y_target, y_pred))
            train_mse_score = np.mean(mse)
            
            ## EPOCH 진행상황 출력
            train_progressbar(step, step_per_epoch
                              , epoch_, EPOCHS
                              ,mean_train_loss ,'mse',train_mse_score , 1, 25)
            step += 1
        
        train_iterator.initialize()
        
        ## auc 
        epoch_time = time.strftime("%H:%M:%S", time.gmtime(timeit.default_timer()-start))
        
        
        val_loss = []
        val_acc = np.array([])
        
        y_pred = np.array([])
        y_target = np.array([])
        mse = []
        
        """ Validation """
        while True:
            try:
                test_images, test_labels = test_iterator.get_item()
                loss_, output_= sess.run([loss, output]
                                         , feed_dict={images: test_images, mask: test_labels
                                       , training: False}) #tf.keras.backend.learning_phase():0
                y_pred = np.append(y_pred, np.array(output_).flatten())
                y_target = np.append(y_target, np.array(test_labels).flatten())
                
                ## losses
                val_loss.append(loss_)
                
                ## mse
                mse.append(mean_squared_error(y_target, y_pred))
                val_mse_score = np.mean(mse)

            except IndexError:
                test_iterator.initialize()
                break
        
        mean_val_loss = np.mean(val_loss)
        
        print(f"\t- val loss : {mean_val_loss:5.5f}\t- val mse : {val_mse_score:5.5f}")
        
        print('time: ',epoch_time, f"\ttrain mse : {train_mse_score:5.5f},\tvalidation mse : {val_mse_score:5.5f}"
              ,'\tbest :', best_score.eval(),  )
        
        # confusion_matrix(y_treu , y_pred)
        print_confusion_matrix(y_target, y_pred, cut_off=0.5)
        
        
        #### end of epoch functions ####
        ## early stop training
        if (epoch_ == 0) or (best_score.eval() > val_mse_score):
            best_score = tf.assign(best_score, val_mse_score)
            patience = 0
        else:
            patience += 1
        print('patience : ', patience)
    
        ## send result to slack
        if SEND_MESSAGE:
            message = "epoch : {} | time : {} \n matrix : {:5.5f}".format(epoch_, epoch_time, confusion_matrix(y_pred>0.5,y_target))
            slack_message('#resnet_project', message)
        
        # save checkpoint
        if SAVE_CHECKPOINT:
            save_path = saver.save(sess, os.path.join(SEGMENT_RESULT_PATH,f'checkpoint_{epoch_}.ckpt'))
        
        # save history 
        history_line = str(epoch_)+','+str(mean_train_loss)+','+str(train_mse_score)+','+ str(mean_val_loss)+','+str(val_mse_score)+'\n'
        if epoch_==1:
            history_line = "epoch,loss,mse,val_loss,val_mse\n" + history_line
        with open(os.path.join(SEGMENT_RESULT_PATH,"history.csv"),'a') as f:
            f.write(history_line)
        
        if EARLY_STOPPING:
            if patience > 7:
                break
            
print('end of line..')
