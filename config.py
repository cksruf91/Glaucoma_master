import os

MY_SLACK_TOKEN = ''

PROJECT = "D:\\Projects\\my_Glaucoma_project_v2"
IMAGE_LOC = os.path.join(PROJECT,"dataset","Glaucoma")
MASK_LOC = os.path.join(PROJECT,"dataset","Glaucoma","mask_od")

TRAIN_IMAGE = os.path.join(IMAGE_LOC,'Training400')
TEST_IMAGE = os.path.join(IMAGE_LOC,'Test')
MASKING_TRAIN_IMAGE = os.path.join(MASK_LOC,'GT','train')
MASKING_VAL_IMAGE = os.path.join(MASK_LOC,'GT','val')

RESULT_PATH = os.path.join(PROJECT, "result")
SEGMENT_RESULT_PATH = os.path.join(PROJECT, "segment_result")

# LABEL
LABEL = {'Glaucoma': 1., 'Non-Glaucoma': 0.}

# Image shape to resahpe the image
# h,w, chennel
IMAGE_SHAPE = (256, 256, 3) # orignal size -(2056, 2124, 3)
OPTIC_DISC_SHAPE = (299, 299, 3)