import sys
import numpy as np
import cv2
import random
import keras
import warnings
import skimage
from skimage.exposure import equalize_adapthist, adjust_gamma
from skimage.transform import rescale, resize, downscale_local_mean, rotate

from config import *
from utils.image_util import image_loader, resize_image, image_rotate, random_gamma, Adaptive_Histogram_Equalization, random_flip_image, normalize_img, crop_optic_disk
from utils.util import print_progress

warnings.filterwarnings(action='ignore') 

class DataGenerator(keras.utils.Sequence):
    def __init__(self, image_dir, masking_dir, batch_size, shape, is_train= True, sample =None):
        self.image_dir = image_dir
        self.masking_dir = masking_dir
        self.batch_size = batch_size
        self.is_train = is_train
        self.shape = shape
        if (sample is not None) & (not isinstance(sample,int)):
            raise ValueError("sample is not integer")
        else:
            self.sample = sample
        self.idx = 0

        self.load_data()
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
            
        self.files = [i for i in zip(image_files, mask_files,name)]
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
#             mask = resize_image(mask,self.shape)
            
            mask = mask[:,:,0]
            mask = np.where(mask<=10,0.0,mask)
            mask = np.where(mask>10,1.0,mask)
            
            mask = mask[:,:,np.newaxis]
            #image = self.argumentation(image,mask,self.is_train)
            data.append((image,mask, name))
        return data
    
    def argumentation(self,img,mask,is_train):
        img = Adaptive_Histogram_Equalization(img)
        
        if is_train:
            img = random_gamma(img)
            ## random한 각도로 image를 회전
            angle = random.randint(1,35)*10
            img = image_rotate(img, angle)
            mask = image_rotate(mask, angle)
            
            ## 50% 확률로 image 뒤집기
            h=random.choice([True,False])
            v=random.choice([True,False])
            img = random_flip_image(img,horizon=h,vertical=v)
            mask = random_flip_image(mask,horizon=h,vertical=v)
        
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
        y_true = np.zeros((self.batch_size,)+self.shape[:2]+(1,)) 
        names = []
        for i,(image,mask,name) in enumerate(temp):
            image,mask = self.argumentation(image,mask,self.is_train)
            x_train[i] = image
            y_true[i] = mask
            names.append(name)
            
        return x_train, y_true
            
    def on_epoch_end(self):
        self.idx = 0
        self.shuffle_item()
    
    def get_label(self):
        if self.is_train:
            raise Exception('train mode cannot return label')
        else :
            labels = []
            for (image,mask,name) in self.data:
                labels.append(mask.tolist())
            return np.array(labels).flatten()