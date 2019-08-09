import json
import os
import numpy as np
import tensorflow as tf
import keras
import keras.backend as K
from keras.callbacks import EarlyStopping, ModelCheckpoint, LearningRateScheduler
from keras.models import model_from_json

from config import *
from model import *
from iterator import DataGenerator
from callback_module import IntervalEvaluation, HistoryCheckpoint, SlackMessage
from utils.util import slack_message

print("tensorflow : ",tf.__version__)
print("keras : ",keras.__version__)

opts = TrainOption('cls')


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

model_json = model.to_json()
with open(os.path.join(RESULT_PATH,'model.json'), 'w') as f:
    f.write(json.dumps(model_json))

with open(os.path.join(RESULT_PATH,'model.json'), 'r') as j:
    model_json = json.loads(j.read())

model = model_from_json(model_json)

def lr_scheduler(epoch):
    lr = opts.LEARNING_RATE
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
adam = keras.optimizers.Adam(0.0)
# sgd = keras.optimizers.SGD(lr=0.045, decay=1e-6, momentum=0.9, nesterov=True)
monitors = auc

## load batch generator
print(f"\ntrain data from : {TRAIN_IMAGE}")
slack_message('#glaucoma', 'train data loading...', MY_SLACK_TOKEN)
train_iterator = DataGenerator(TRAIN_IMAGE, MASK_LOC ,opts.BATCH_SIZE
                               ,IMAGE_SHAPE, OPTIC_DISC_SHAPE, is_train=True, copy = True, sample=None)

print(f"\ntest data from : {TEST_IMAGE}")
slack_message('#glaucoma', 'test data loading...', MY_SLACK_TOKEN)
test_iterator = DataGenerator(TEST_IMAGE, MASK_LOC ,opts.BATCH_SIZE 
                               ,IMAGE_SHAPE, OPTIC_DISC_SHAPE, is_train=False, copy = False, sample=None)

call_backs = [
    IntervalEvaluation(test_iterator, loss_func, monitor_name = monitors.__name__),
    EarlyStopping(monitor=f'val_{monitors.__name__}', patience =5, verbose =1 , mode ='max'),
    ModelCheckpoint(os.path.join(RESULT_PATH, "checkpoint-{epoch:03d}.h5"),
                    monitor=f'val_{monitors.__name__}', save_best_only=False, mode='max'),
    LearningRateScheduler(lr_scheduler, verbose=1),
    HistoryCheckpoint(os.path.join(RESULT_PATH, "checkpoint_hist.csv"), monitors.__name__),
    SlackMessage(MY_SLACK_TOKEN,monitors.__name__)
]

model.compile(loss = loss_func, optimizer = adam, metrics = [monitors,sensitivity,specificity])
model.summary()

try:
    checkpoint ='checkpoint-024.h5'
    init_epoch = int(checkpoint.split("-")[-1].split(".")[0])
    model.load_weights(os.path.join(RESULT_PATH,checkpoint))
    print("*******************\ncheckpoint restored\n*******************")
    slack_message('#glaucoma', f'checkpoint restored : {checkpoint}', MY_SLACK_TOKEN)
except:
    init_epoch = 0
    print("*******************\nfailed to load checkpoint\n*******************")
    slack_message('#glaucoma', 'failed to load checkpoint', MY_SLACK_TOKEN)


slack_message('#glaucoma', f'start learing rate {opts.LEARNING_RATE}', MY_SLACK_TOKEN)

with tf.device('/device:GPU:0'):
    hist = model.fit_generator(generator=train_iterator,
                               steps_per_epoch=None,
                               epochs=40,
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
print(hist)
