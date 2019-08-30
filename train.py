import json
import os
import argparse
import numpy as np
import tensorflow as tf
import keras
import keras.backend as K
from keras.callbacks import EarlyStopping, ModelCheckpoint, LearningRateScheduler
from keras.models import model_from_json

from config import *
from model import *
from iterator import DataIterator
from callback_module import IntervalEvaluation, HistoryCheckpoint, SlackMessage
from utils.util import slack_message, last_cheackpoint, get_config

def args():
    parser = argparse.ArgumentParser()
    parser.add_argument('-t', '--test',action="store_true", help='test mode')  # number of class
    args = parser.parse_args()
    return args

testmode = 10 if args().test else None

print("tensorflow : ",tf.__version__)
print("keras : ",keras.__version__)


# from keras.applications.xception import Xception
# """use pretrained"""
# with tf.device('/device:GPU:0'):
#     model = Xception() #weights='imagenet'
#     model.layers.pop()
#     inputs = model.layers[0].output
#     x= model.layers[-1].output
#     out= keras.layers.Dense(units=1, activation='sigmoid',kernel_initializer ='he_normal')(x) 
#     model = keras.Model(inputs=inputs, outputs=out)

with tf.device('/device:GPU:0'):
    xception = Xception()
    model = xception.build(OPTIC_DISC_SHAPE)
#     model = ResNetV3(True).build(OPTIC_DISC_SHAPE)

# model_json = model.to_json()
# with open(os.path.join(RESULT_PATH,'model.json'), 'w') as f:
#     f.write(json.dumps(model_json))

with open(os.path.join(RESULT_PATH,'model.json'), 'r') as j:
    model_json = json.loads(j.read())

model = model_from_json(model_json)

def lr_scheduler(epoch):
    lr = 1e-4
    t = 0.94**(epoch)
    new_lr =  round(lr*t,7)
    return max(new_lr,1e-08)

""" metrics functions """
def auc(y_true, y_pred):
    y_pred = tf.clip_by_value(y_pred,0,1)
    auc = tf.metrics.auc(y_true, y_pred, curve = 'PR',summation_method = 'careful_interpolation')[1]
    K.get_session().run(tf.local_variables_initializer())
    return auc

def sensitivity(y_true, y_pred):
    true_positives = K.sum(K.round(K.clip(y_true * y_pred, 0, 1)))
    possible_positives = K.sum(K.round(K.clip(y_true, 0, 1)))
    return true_positives / (possible_positives + K.epsilon())

def specificity(y_true, y_pred):
    true_negatives = K.sum(K.round(K.clip((1-y_true) * (1-y_pred), 0, 1)))
    possible_negatives = K.sum(K.round(K.clip(1-y_true, 0, 1)))
    return true_negatives / (possible_negatives + K.epsilon())


loss_func = 'binary_crossentropy'#'hinge'
optim = keras.optimizers.Adam(0.0)
# sgd = keras.optimizers.SGD(lr=0.045, decay=1e-6, momentum=0.9, nesterov=True)
monitors = auc
BATCH_SIZE = 1

model.compile(loss = loss_func, optimizer = optim, metrics = [monitors,sensitivity,specificity])
model.summary()

augm = {"gamma":True, "rotate":True, "polar":True, "hiseq":True, "normal":True, "flip":True, "copy":True}
## load batch generator
print(f"\ntrain data from : {TRAIN_IMAGE}")
train_iterator = DataIterator(TRAIN_IMAGE, MASK_LOC, BATCH_SIZE, IMAGE_SHAPE, OPTIC_DISC_SHAPE
                              , is_train=True, sample=testmode
                              , copy = augm['copy'], rotate = augm['rotate'], polar = augm['polar'], hiseq = augm['hiseq']
                              , gamma = augm['gamma'], flip = augm['flip'], normal = augm['normal'])

print(f"\ntest data from : {TEST_IMAGE}")
test_iterator = DataIterator(TEST_IMAGE, MASK_LOC, BATCH_SIZE, IMAGE_SHAPE, OPTIC_DISC_SHAPE
                             , is_train=False, sample=testmode
                             , polar= augm['polar'], hiseq = augm['hiseq'], normal = augm['normal'])

call_backs = [
    IntervalEvaluation(test_iterator, loss_func, monitor_name = monitors.__name__),
    EarlyStopping(monitor=f'val_{monitors.__name__}', patience =5, verbose =1 , mode ='max'),
#     ModelCheckpoint(os.path.join(RESULT_PATH, "checkpoint-{epoch:03d}.h5"),
#                     monitor=f'val_{monitors.__name__}', save_best_only=False, mode='max'),
    LearningRateScheduler(lr_scheduler, verbose=1),
    HistoryCheckpoint(os.path.join(RESULT_PATH, "checkpoint_hist.csv"), monitors.__name__),
#     SlackMessage(MY_SLACK_TOKEN,monitors.__name__)
]


try:
    weight = last_cheackpoint(RESULT_PATH)
    init_epoch = int(weight.split("-")[-1].split(".")[0])
    model.load_weights(os.path.join(RESULT_PATH,weight))
    print("*******************\ncheckpoint restored\n*******************")
#     slack_message('#glaucoma', f'checkpoint restored : {weight}', MY_SLACK_TOKEN)
except:
    init_epoch = 0
    print("*******************\nfailed to load checkpoint\n*******************")
#     slack_message('#glaucoma', 'failed to load checkpoint', MY_SLACK_TOKEN)


train_options = {"optimizer":get_config(optim), "batchsize":BATCH_SIZE, "loss_function":loss_func
                 , "input_shape":OPTIC_DISC_SHAPE, "augmemtation":augm}

print(json.dumps(train_options, indent=4, sort_keys=False))

with open(os.path.join(RESULT_PATH,'train_options.json'),'w') as f:
    f.write(json.dumps(train_options))

with tf.device('/device:GPU:0'):
    hist = model.fit_generator(generator=train_iterator,
                               steps_per_epoch=None,
                               epochs=100,
                               verbose=1,
                               callbacks=call_backs,
                               class_weight=None,
                               max_queue_size=10,
                               workers=4,
                               use_multiprocessing=False,
                               initial_epoch=init_epoch,
                               shuffle =False
                               # validation_data=test_iterator,
                               # validation_steps=None,
                               )
print(hist.history)
