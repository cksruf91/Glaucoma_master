import numpy as np
import keras
import os
import sys
import json
import sys
import argparse
import skimage

from utils.image_util import *
from utils.util import pbar, last_cheackpoint
from config import *

def args():
    parser = argparse.ArgumentParser()
    parser.add_argument('-m','--mode', type=int, required=True, choices=[0,1]
                        , help='1: train image, 0: test image')  # number of class
    args = parser.parse_args()
    return args

arg = args()

## 1 : train image 2: test image
if arg.mode:
    image_loc = TRAIN_IMAGE
else :
    image_loc = TEST_IMAGE

## Model Loading
with open(os.path.join(SEGMENT_RESULT_PATH,'model.json'), 'r') as f:
    model_json = json.loads(f.read())

model = keras.models.model_from_json(model_json)
weight = last_cheackpoint(SEGMENT_RESULT_PATH)
print(f"restore check point : {os.path.basename(weight)}")
model.load_weights(weight)


## image Loading
image_files = []
for (path, dir, files) in os.walk(image_loc):
    if path ==image_loc: # 현재 디렉토리는 넘김
        continue
    for file in files:
        image_files.append(os.path.join(path, file))

        
## number of image 
n_image = len(image_files)

## train 했을때의 옵션
with open(os.path.join(SEGMENT_RESULT_PATH,'train_options.json'), 'r') as f:
    train_options = json.loads(f.read())

hiseq = train_options['augmemtation']['hiseq']
normal = train_options['augmemtation']['normal']

data = np.zeros((n_image,) +IMAGE_SHAPE)
for i, file in enumerate(pbar(image_files, prefix="loading...")):
    #print_progress(n_image, i+1, "loading...")
    
    image = image_loader(file)
    image = resize_image(image,IMAGE_SHAPE)
    
    ## Equalize and normalize if you do it when training
    image = Adaptive_Histogram_Equalization(image) if hiseq else image
    image = normalize_img(image) if normal else image
    
    data[i] = image

## prediction
print("predict..")
y_pred = model.predict(data, verbose=1,batch_size=1)
y_pred = np.argmax(y_pred,-1)

## 0~1 -> 0 ~ 255
y_pred = np.where(y_pred>=0.3,255,y_pred)
y_pred = np.where(y_pred<0.3,0,y_pred)

## chennel
# mask = np.zeros((n_image,)+IMAGE_SHAPE)

opensize = 10
for idx in pbar(range(n_image), prefix="save..."):
    ## remove noise
    opened = opening_image(y_pred[idx], opensize)
    closed = closing_image(opened, opensize)
    
    ## generate file name
    f = "mask_" + os.path.basename(image_files[idx])
    f = os.path.join(MASK_LOC, f)
    
    #print_progress(n_image, idx+1, "saveing..")
    
    mask = np.zeros(IMAGE_SHAPE)
    for i in range(3):
        mask[:,:,i] = closed

    ## save
    skimage.io.imsave(f, mask.astype(np.uint8))
