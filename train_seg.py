import json
import os
import argparse
import tensorflow as tf
import keras
import keras.backend as K
from keras.callbacks import EarlyStopping, ModelCheckpoint, LearningRateScheduler

from config import *
from model import *
from seg_iterator import DataIterator
from utils.util import last_cheackpoint, get_config
from callback_module import IntervalEvaluation, HistoryCheckpoint, SlackMessage

print("tensorflow : ",tf.__version__)
print("keras : ",keras.__version__)

""" metrics functions """
def auc(y_true, y_pred):
    auc = tf.metrics.auc(y_true, y_pred, curve = 'PR',summation_method = 'careful_interpolation')[1]
    K.get_session().run(tf.local_variables_initializer())
    return auc

def args():
    parser = argparse.ArgumentParser()
    parser.add_argument('-t', '--test',action="store_true", help='test mode')  # number of class
    args = parser.parse_args()
    return args

monitors = 'mse'
loss_func = 'categorical_crossentropy' #binary_crossentropy
testmode = 10 if args().test else None
BATCH_SIZE = 1

optim = keras.optimizers.Adam(lr=1e-3, beta_1=0.9, beta_2=0.999, epsilon=None, decay=0.0, amsgrad=False)
# optim = keras.optimizers.Adadelta(lr=1.0, rho=0.95, epsilon=None, decay=0.9)
# optim = keras.optimizers.SGD(lr=0.045, decay=1e-6, momentum=0.9, nesterov=True)
# optim = keras.optimizers.RMSprop(lr=0.001, rho=0.9, epsilon=None, decay=0.0)
# optim = keras.optimizers.Nadam(lr=0.001, beta_1=0.9, beta_2=0.999, epsilon=None, schedule_decay=0.004)
learning_rate = optim.get_config()['lr']
def lr_scheduler(epoch):
    lr = learning_rate
    new_lr = lr * 0.1**(epoch//10)
    return max(new_lr,1e-7)

with tf.device('/device:GPU:0'):
    unet = Unet().build(IMAGE_SHAPE)
model_json = unet.to_json()

with open(os.path.join(SEGMENT_RESULT_PATH,'model.json'), 'w') as f:
    f.write(json.dumps(model_json))

unet.compile(loss = loss_func, optimizer = optim, metrics = [monitors])
unet.summary()

# with open(os.path.join(SEGMENT_RESULT_PATH,'model.json'), 'r') as f:
#     model_json = json.loads(f.read())
# unet = keras.models.model_from_json(model_json)

augm = {"gamma":True, "rotate":True, "flip":True, "hiseq":False, "normal":True, "invert":False, "crop":True}

## load batch generator
print(f"\ntrain data from : {MASKING_TRAIN_IMAGE}")
train_iterator = DataIterator(TRAIN_IMAGE, MASKING_TRAIN_IMAGE, BATCH_SIZE, IMAGE_SHAPE, is_train=True, sample = testmode
                            , gamma=augm["gamma"], rotate=augm["rotate"], flip=augm["flip"]
                              , hiseq=augm["hiseq"], normal=augm["normal"], invert=augm["invert"], crop = augm["crop"])

print(f"\ntest data from : {MASKING_VAL_IMAGE}")
test_iterator = DataIterator(TRAIN_IMAGE, MASKING_VAL_IMAGE, BATCH_SIZE, IMAGE_SHAPE, is_train=False,sample = testmode
                            , hiseq=augm["hiseq"], normal=augm["normal"])


call_backs = [
    IntervalEvaluation(test_iterator, loss_func, monitor_name = monitors),
    EarlyStopping(monitor=f'val_{monitors}', patience =10, verbose =1 , mode ='min'),
    ModelCheckpoint(os.path.join(SEGMENT_RESULT_PATH, "checkpoint-{epoch:03d}.h5"),
                    monitor=f'val_{monitors}', save_best_only=True, mode='min'),
    LearningRateScheduler(lr_scheduler, verbose=1),
    HistoryCheckpoint(os.path.join(SEGMENT_RESULT_PATH, "checkpoint_hist.csv"), monitors),
#     SlackMessage(MY_SLACK_TOKEN,monitors)
]

try:
    weight = last_cheackpoint(SEGMENT_RESULT_PATH)
    print(weight)
    init_epoch = int(os.path.basename(weight.split("-")[-1].split(".")[0]))
    unet.load_weights(weight)
    print("*******************\ncheckpoint restored\n*******************")
except:
    init_epoch = 0
    print("*******************\nfailed to load checkpoint\n*******************")

train_options = {"optimizer":get_config(optim), "batchsize":BATCH_SIZE, "loss_function":loss_func
                 , "input_shape":IMAGE_SHAPE, "augmemtation":augm}
print(json.dumps(train_options, indent=4, sort_keys=False))
with open(os.path.join(SEGMENT_RESULT_PATH,'train_options.json'),'w') as f:
    f.write(json.dumps(train_options))

""" run train """
hist = unet.fit_generator(generator=train_iterator,
                    steps_per_epoch=None,
                    epochs=60,
                    verbose=1,
                    callbacks=call_backs,
                    class_weight=None,
                    max_queue_size=10,
                    workers=1,
                    use_multiprocessing=False,
                    initial_epoch=init_epoch
                    #validation_data=test_iterator,
                    #validation_steps=None,
                   )
print(hist.history)
