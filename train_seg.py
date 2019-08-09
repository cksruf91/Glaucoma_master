import json
import os
import tensorflow as tf
import keras
import keras.backend as K
from keras.callbacks import EarlyStopping, ModelCheckpoint, LearningRateScheduler

from config import *
from model import *
from seg_iterator import DataGenerator
from callback_module import IntervalEvaluation, HistoryCheckpoint

print("tensorflow : ",tf.__version__)
print("keras : ",keras.__version__)

opts = TrainOption('seg')

with tf.device('/device:GPU:0'):
    unet = Unet(True).build(IMAGE_SHAPE)

model_json = unet.to_json()
with open(os.path.join(SEGMENT_RESULT_PATH,'model.json'), 'w') as f:
    f.write(json.dumps(model_json))
    
def lr_scheduler(epoch,lr):    
    new_lr = lr* round(0.1**(epoch//5),7)
    return max(new_lr, 1e-10)

""" metrics functions """
def auc(y_true, y_pred):
    auc = tf.metrics.auc(y_true, y_pred, curve = 'PR',summation_method = 'careful_interpolation')[1]
    K.get_session().run(tf.local_variables_initializer())
    return auc

loss_func = 'binary_crossentropy'
adam = keras.optimizers.Adam(opts.LEARNING_RATE)
monitors = 'mse'

## load batch generator
print(f"\ntrain data from : {MASKING_TRAIN_IMAGE}")
train_iterator = DataGenerator(TRAIN_IMAGE, MASKING_TRAIN_IMAGE
                               ,opts.BATCH_SIZE
                               ,IMAGE_SHAPE, is_train=True,sample = None)

print(f"\ntest data from : {MASKING_VAL_IMAGE}")
test_iterator = DataGenerator(TRAIN_IMAGE, MASKING_VAL_IMAGE
                               ,opts.BATCH_SIZE
                               ,IMAGE_SHAPE, is_train=False,sample = None)


call_backs = [
    IntervalEvaluation(test_iterator, monitor_name = monitors),
    EarlyStopping(monitor=f'val_{monitors}', patience =7, verbose =1 , mode ='min'),
    ModelCheckpoint(os.path.join(SEGMENT_RESULT_PATH, "checkpoint-{epoch:03d}.h5"),
                    monitor=f'val_{monitors}', save_best_only=True, mode='min'),
    LearningRateScheduler(lr_scheduler, verbose=1),
    HistoryCheckpoint(os.path.join(SEGMENT_RESULT_PATH, "checkpoint_hist.csv"), monitors)
]

unet.compile(loss = loss_func, optimizer = adam, metrics = [monitors])
unet.summary()

try:
    checkpoint ='checkpoint-025.h5'
    init_epoch = int(checkpoint.split("-")[-1].split(".")[0])
    unet.load_weights(os.path.join(SEGMENT_RESULT_PATH,checkpoint))
    print("checkpoint restored")
except:
    init_epoch = 0
    print("failed to load checkpoint")

""" run train """
hist = unet.fit_generator(generator=train_iterator,
                    steps_per_epoch=None,
                    epochs=100,
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
print(hist)
