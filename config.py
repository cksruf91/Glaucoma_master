import os

MY_SLACK_TOKEN = ''

PROJECT = "D:\\Projects\\my_Glaucoma_project_v2"
IMAGE_LOC = os.path.join(PROJECT,"dataset\\Glaucoma")
MASK_LOC = os.path.join(PROJECT,"dataset\\Glaucoma\\mask_od")

TRAIN_IMAGE = os.path.join(IMAGE_LOC,'Training400')
TEST_IMAGE = os.path.join(IMAGE_LOC,'Test')
MASKING_TRAIN_IMAGE = os.path.join(MASK_LOC,'GT\\train')
MASKING_VAL_IMAGE = os.path.join(MASK_LOC,'GT\\val')

# TRAIN_FILE = os.path.join(IMAGE_LOC,'train.tfrecord')
# TEST_FILE = os.path.join(IMAGE_LOC,'test.tfrecord')
RESULT_PATH = os.path.join(PROJECT, "result")

# SEGMENT_TRAIN_FILE = os.path.join(IMAGE_LOC,'segment_train.tfrecord')
# SEGMENT_TEST_FILE = os.path.join(IMAGE_LOC,'segment_test.tfrecord')
SEGMENT_RESULT_PATH = os.path.join(PROJECT, "segment_result")

# LABEL
LABEL = {'Glaucoma': 1., 'Non-Glaucoma': 0.}

# Image shape to resahpe the image
# h,w, chennel
IMAGE_SHAPE = (512, 512, 3) # orignal size -(2056, 2124, 3)
OPTIC_DISC_SHAPE = (256, 256, 3)

SEND_MESSAGE = False
SAVE_CHECKPOINT = True
EARLY_STOPPING = False
SHUFFLE_BUFFER = 50


# training options
class TrainOption():
    def __init__(self,train_type):
        if train_type =='cls':
            self.LEARNING_RATE = 1e-2
            self.LR_DEACY_STEPS = 2000
            self.LR_DECAY_RATE = 0.96
            self.MOMENTUM = 0.9
            self.WEIGHT_DECAY = 0.0001
            self.BATCH_SIZE = 2
            self.EPOCHS = 100
            self.STEP_PER_EPOCH = None #1562
            self.DROPOUT_RATE = 0.0
        elif train_type =='seg':
            self.LEARNING_RATE = 1e-3
            self.LR_DEACY_STEPS = 2000
            self.LR_DECAY_RATE = 0.96
            self.MOMENTUM = 0.9
            self.WEIGHT_DECAY = 0.0001
            self.BATCH_SIZE = 1
            self.EPOCHS = 1000
            self.STEP_PER_EPOCH = None #1562
            self.DROPOUT_RATE = 0.0
        else:
            raise ValueError('TrainOption : invalied train type [cls,seg]')
