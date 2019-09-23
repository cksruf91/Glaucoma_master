import sys
import numpy as np
import cv2
import random
import keras 
import warnings
import skimage
import h5py
from skimage.exposure import equalize_adapthist, adjust_gamma
from skimage.transform import rescale, resize, downscale_local_mean, rotate

from config import *
from utils.image_util import image_loader, resize_image, image_rotate, random_gamma, Adaptive_Histogram_Equalization, random_flip_image, normalize_img, crop_optic_disk, polartransform_image, random_invert_image
from utils.util import print_progress

warnings.filterwarnings(action='ignore') 
        
class DataIterator(keras.utils.Sequence):
    def __init__(self, infile, batch_size, input_shape
                 , is_train= True, copy=False, rotate=False, polar=False, hiseq=False
                 ,get_roi=False, gamma=False, flip=False, normal=False, invert=False):
        self.image = h5py.File(infile)['image']
        self.mask = h5py.File(infile)['mask']
        self.label = h5py.File(infile)['label']
        self.list_ids = list(range(self.image.shape[0]))
        self.data_length = len(self.list_ids)
        
        self.batch_size = batch_size
        self.is_train = is_train
        self.input_shape = input_shape
        
        self.rotate = rotate
        self.polar = polar
        self.hiseq = hiseq
        self.get_roi = get_roi
        self.gamma = gamma
        self.flip = flip
        self.normal = normal
        self.invert = invert
        
        self.idx = 0

        if self.is_train:
            self.shuffle_item()
        
    def __len__(self):
        return int(self.data_length / self.batch_size)
    
    def augmentation(self,img,mask,is_train):
        if is_train:
            ## random한 각도로 image를 회전
            angle = random.randint(1,35)*10
            img = image_rotate(img, angle) if self.rotate else img 
            # mask = image_rotate(mask, angle) if self.rotate else mask
        
        ## get optic disk
        # img = crop_optic_disk(img, mask, margin=3) if self.get_roi else img
        
        ## Polar Transform
        # angle = random.randint(1, 35)*10
        # img = polartransform_image(img, angle) if self.polar else img
        
        ## Adaptive_Histogram_Equalization
        img = Adaptive_Histogram_Equalization(img, cl=0.03) if self.hiseq else img
        
        if is_train:
            ## random gamma
            img = random_gamma(img) if self.gamma else img 
            
            ## 50% 확률로 이미지 상하 혹은 좌우 반전
            h=random.choice([True, False])
            v=random.choice([True, False])
            img = random_flip_image(img, horizon = h, vertical = v) if self.flip else img 
            ## image invert
            img = random_invert_image(img) if self.invert else img
        
        ## image 정규화
        img = normalize_img(img) if self.normal else img 
        
        return resize_image(img,self.input_shape)

    def shuffle_item(self):
        random.shuffle(self.list_ids)

    def get_batch(self,i):
        from_ = i*self.batch_size
        to_ = (i+1)*self.batch_size
        
        if self.data_length < to_:
            raise IndexError("index out of range")
        
        temp_ids = self.list_ids[from_:to_]
        return zip(self.image[temp_ids], self.mask[temp_ids], self.label[temp_ids])
    
    def __getitem__(self,index): #get_item
        temp = self.get_batch(index) #self.idx
        self.idx += 1

        x_train = np.zeros((self.batch_size,)+self.input_shape)
        y_true = np.zeros((self.batch_size,1))
        names = []
        for i,(image, mask, label) in enumerate(temp):
            image = self.augmentation(image,mask,self.is_train)
            x_train[i] = image
            y_true[i] = label
            # names.append(name)

        return x_train, y_true
            
    def on_epoch_end(self): #initialize
        self.idx = 0
        if self.is_train:
            self.shuffle_item()
        
    def get_label(self):
        if self.is_train:
            raise Exception('train mode cannot return label')
        else :
            labels = self.label[self.list_ids]
            return np.array(labels)
        