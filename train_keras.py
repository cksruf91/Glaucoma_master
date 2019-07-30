import json
import os
import tensorflow as tf
import keras
import keras.backend as K
from keras.callbacks import EarlyStopping, ModelCheckpoint, LearningRateScheduler

from config import *
from model import *
from callback_module import IntervalEvaluation
from iterator import DataGenerator

print("tensorflow : ",tf.__version__)
print("keras : ",keras.__version__)

opts = TrainOption('cls')

model = ResNetV3(True).build(OPTIC_DISC_SHAPE)

model_json = model.to_json()
with open(os.path.join(RESULT_PATH,'model.json'), 'w') as f:
    f.write(json.dumps(model_json))

# with open(os.path.join(RESULT_PATH,'model.json'), 'r') as j:
#     model_json = json.loads(j.read())
# from keras.models import model_from_json
# model = model_from_json(model_json)
# model.summary()

def lr_scheduler(epoch,lr=opts.LEARNING_RATE):    
    new_lr = lr* round(0.1**(epoch//5),7)
    return lr

""" metrics functions """
def auc(y_true, y_pred):
    auc = tf.metrics.auc(y_true, y_pred, curve = 'PR',summation_method = 'careful_interpolation')[1]
    K.get_session().run(tf.local_variables_initializer())
    return auc

loss_func = 'binary_crossentropy'
adam = keras.optimizers.Adam(0.0)
monitors = auc

## load batch generator
print(f"\ntrain data from : {TRAIN_IMAGE}")
train_iterator = DataGenerator(TRAIN_IMAGE, MASK_LOC ,opts.BATCH_SIZE
                               ,IMAGE_SHAPE, OPTIC_DISC_SHAPE, is_train=True, copy = True, sample=None)

print(f"\ntest data from : {TEST_IMAGE}")
test_iterator = DataGenerator(TEST_IMAGE, MASK_LOC ,opts.BATCH_SIZE 
                               ,IMAGE_SHAPE, OPTIC_DISC_SHAPE, is_train=False, copy = False, sample=None)

call_backs = [
    IntervalEvaluation(test_iterator, monitor_name = monitors.__name__),
    EarlyStopping(monitor=f'val_{monitors.__name__}', patience =5, verbose =1 , mode ='max'),
    ModelCheckpoint(os.path.join(RESULT_PATH, "checkpoint-{epoch:03d}.h5"),
                    monitor=f'val_{monitors.__name__}', save_best_only=False, mode='max'),
    LearningRateScheduler(lr_scheduler, verbose=1)
#     HistoryCheckpoint(os.path.join(RESULT_PATH, "checkpoint_hist.csv"), monitors.__name__)
]

model.compile(loss = loss_func, optimizer = adam, metrics = [monitors])
model.summary()

hist = model.fit_generator(generator=train_iterator,
                    steps_per_epoch=None,
                    epochs=100,
                    verbose=1,
                    callbacks=call_backs,
                    class_weight=None,
                    max_queue_size=10,
                    workers=1,
                    use_multiprocessing=False,
                    initial_epoch=0
                    #validation_data=test_iterator,
                    #validation_steps=None,
                   )
print(hist)