import sys
import numpy as np
import cv2
import random
import keras
import warnings
import skimage
import collections
from skimage.exposure import equalize_adapthist, adjust_gamma
from skimage.transform import rescale, resize, downscale_local_mean, rotate

from keras.utils import to_categorical

from config import *
from utils.image_util import image_loader, resize_image, image_rotate, random_gamma, Adaptive_Histogram_Equalization, random_flip_image, normalize_img, crop_optic_disk, random_invert_image, per_chenel_normalize, random_crop
from utils.util import print_progress

warnings.filterwarnings(action='ignore') 

class DataIterator(keras.utils.Sequence):
    def __init__(self, image_dir, masking_dir, batch_size, shape, 
                 is_train=True, sample=None, gamma=False, rotate=False, flip=False, hiseq=False, normal=False, invert=False,
                 crop = False):
        self.image_dir = image_dir
        self.masking_dir = masking_dir
        self.batch_size = batch_size
        self.is_train = is_train
        self.shape = shape
        self.gamma = gamma
        self.rotate = rotate
        self.flip = flip
        self.hiseq = hiseq
        self.normal = normal
        self.invert = invert
        self.crop = crop
        if (sample is not None) & (not isinstance(sample,int)):
            raise ValueError("sample is not integer")
        else:
            self.sample = sample
        self.idx = 0

        self.load_data()
        if self.is_train:
            self.shuffle_item()
        
    def __len__(self):
        return int(self.data_length/self.batch_size)
    
    def load_data(self):
        image_files = []
        name = []
        mask_files = []
        for f in os.listdir(self.masking_dir):
            file_name = os.path.join(self.masking_dir,f)
            mask_files.append(file_name)
            
            for (path, di, files) in os.walk(self.image_dir):
                if path ==self.image_dir: # 현재 디렉토리는 넘김
                    continue
                for file in files:
                    f = f.replace("mask_","")
                    if f == file:
                        image_files.append(os.path.join(path, file))
                        name.append(file)
            
        self.files = [i for i in zip(image_files, mask_files, name)]
        random.shuffle(self.files)  
        self.total_length = len(self.files)
        
        if isinstance(self.sample,int):
            self.files = self.files[:self.sample]
        self.data = self.preprocess(self.files)
        self.data_length = len(self.data)
    
    def preprocess(self,files):
        data = []
        for i,(img, mask, name) in enumerate(files):
            print_progress(self.total_length,i+1)
            image = image_loader(img)
            mask = image_loader(mask)
            
            ## resize
            image = resize_image(image,self.shape)
            mask = resize_image(mask,self.shape)
            
            mask = mask[:,:,0]
            mask = np.where(mask<=0.3,0.0,mask)
            mask = np.where(mask>0.3,1.0,mask)
            
            mask = to_categorical(mask,2)

            data.append((image,mask, name))
        return data
    
    def augmentation(self,img,mask,is_train):
        
        if is_train:
            img = random_gamma(img) if self.gamma else img 
            ## random한 각도로 image를 회전
            angle = random.randint(1,35)*10
            img = image_rotate(img, angle) if self.rotate else img 
            mask = image_rotate(mask, angle) if self.rotate else mask 
            
            ## 50% 확률로 image 뒤집기
            h=random.choice([True,False])
            v=random.choice([True,False])
            img = random_flip_image(img,horizon=h,vertical=v) if self.flip else img 
            mask = random_flip_image(mask,horizon=h,vertical=v) if self.flip else mask 
            
            ## image invert
            img = random_invert_image(img) if self.invert else img

            ## image crop
            if self.crop:
                img, mask = random_crop(img, mask, 0.2)
            
        
        img = Adaptive_Histogram_Equalization(img) if self.hiseq else img 
        img = normalize_img(img) if self.normal else img 
        return img ,mask

    def shuffle_item(self):
        random.shuffle(self.data)

    def get_batch(self,i):
        from_ = i*self.batch_size
        to_ = (i+1)*self.batch_size
        if self.data_length < to_:
            raise IndexError("index out of range")
        return self.data[from_:to_]
    
    def __getitem__(self,index):
        
        temp = self.get_batch(index)
        self.idx += 1

        x_train = np.zeros((self.batch_size,)+self.shape)
        y_true = np.zeros((self.batch_size,)+self.shape[:2]+(2,)) 
        names = []
        for i,(image,mask,name) in enumerate(temp):
            
            image,mask = self.augmentation(image,mask,self.is_train)
            x_train[i] = image
            y_true[i] = mask
            names.append(name)

        return x_train, y_true
            
    def on_epoch_end(self):
        self.idx = 0
        if self.is_train:
            self.shuffle_item()
    
    def get_label(self):
        if self.is_train:
            raise Exception('train mode cannot return label')
        else :
            labels = np.zeros((self.data_length,)+self.shape[:2]+(2,))
            for i,(image,mask,name) in enumerate(self.data):
                labels[i] = mask
            return labels
        
