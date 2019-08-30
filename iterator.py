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
from utils.image_util import image_loader, resize_image, image_rotate, random_gamma, Adaptive_Histogram_Equalization, random_flip_image, normalize_img, crop_optic_disk, polartransform_image, random_invert_image
from utils.util import print_progress

warnings.filterwarnings(action='ignore') 
        
class DataIterator(keras.utils.Sequence):
    def __init__(self, image_dir, masking_dir,batch_size, load_shape, disc_shape
                 , is_train= True, copy=False, sample =None, rotate=False, polar=False, hiseq=False
                 , gamma=False, flip=False, normal=False, invert=False):
        self.image_dir = image_dir
        self.masking_dir = masking_dir
        self.batch_size = batch_size
        self.is_train = is_train
        self.load_shape = load_shape
        self.disc_shape = disc_shape
        self.rotate = rotate
        self.polar = polar
        self.hiseq = hiseq
        self.gamma = gamma
        self.flip = flip
        self.normal = normal
        self.invert = invert
        
        if (sample is not None) & (not isinstance(sample,int)):
            raise ValueError("sample is not integer")
        else:
            self.sample = sample
        self.idx = 0

        self.load_data(copy)
        if self.is_train:
            self.shuffle_item()
        
    
    def __len__(self):
        return int(self.data_length/self.batch_size)
    
    def load_data(self,copy):
        image_files = []
        file_name = []
        for (path, dir, files) in os.walk(self.image_dir):
            if path ==self.image_dir: # 현재 디렉토리는 넘김
                continue
            numcopy = 9 if (copy and path.split('\\')[-1] == "Glaucoma") else 1
            for file in files*numcopy:
                image_files.append(os.path.join(path, file))
                file_name.append(file)
        
        mask_files = []
        labels = []
        
        for f in image_files:
            label = os.path.dirname(f).split('\\')[-1]
            labels.append(LABEL[label])
            m = 'mask_'+os.path.basename(f)
            m = os.path.join(self.masking_dir,m)
            mask_files.append(m)
            
        self.files = [i for i in zip(image_files, mask_files, labels,file_name)]
        random.shuffle(self.files)  
        self.total_length = len(self.files)
        
        if isinstance(self.sample,int):
            self.files = self.files[:self.sample]
        self.data = self.preprocess(self.files)
        self.data_length = len(self.data)
    
    def preprocess(self,files):
        data = []
        for i,(img,mask,label,name) in enumerate(files):
            print_progress(self.total_length,i+1)
            image = image_loader(img)
            mask = image_loader(mask)
            ## resize
            image = resize_image(image,self.load_shape)
            mask = resize_image(mask,self.load_shape)
            
            mask = mask[:,:,0]
            mask = np.where(mask<=0.3,0.0,mask)
            mask = np.where(mask>0.3,1.0,mask)
            mask = mask[:,:,np.newaxis]
            #mask = resize_image(mask,self.load_shape)
            #image = self.argumentation(image,mask,self.is_train)
            data.append((image,mask,label,name))
        return data
    
    def augmentation(self,img,mask,is_train):
        if is_train:
            ## random한 각도로 image를 회전
            angle = random.randint(1,35)*10
            img = image_rotate(img, angle) if self.rotate else img 
            mask = image_rotate(mask, angle) if self.rotate else mask
        ## get optic disk
        img = crop_optic_disk(img,mask,margin=3)
        ## Polar Transform
        angle = random.randint(1,35)*10
        img = polartransform_image(img,angle) if self.polar else img
        ## Adaptive_Histogram_Equalization
        img = Adaptive_Histogram_Equalization(img,cl=0.03) if self.hiseq else img
        
        if is_train:
            ## random gamma
            img = random_gamma(img) if self.gamma else img 
            ## 50% 확률로 이미지 상하 혹은 좌우 반전
            h=random.choice([True,False])
            v=random.choice([True,False])
            img = random_flip_image(img,horizon=h,vertical=v) if self.flip else img 
            ## image invert
            img = random_invert_image(img) if self.invert else img
        
        ## image 정규화
        img = normalize_img(img) if self.normal else img 
        
        return resize_image(img,self.disc_shape)

    def shuffle_item(self):
        random.shuffle(self.data)

    def get_batch(self,i):
        from_ = i*self.batch_size
        to_ = (i+1)*self.batch_size
        if self.data_length < to_:
            raise IndexError("index out of range")
        return self.data[from_:to_]
    
    def __getitem__(self,index): #get_item
        temp = self.get_batch(index) #self.idx
        self.idx += 1

        x_train = np.zeros((self.batch_size,)+self.disc_shape)
        y_true = np.zeros((self.batch_size,1))
        names = []
        for i,(image,mask,label,name) in enumerate(temp):
            image = self.augmentation(image,mask,self.is_train)
            x_train[i] = image
            y_true[i] = label
            names.append(name)

        return x_train, y_true
            
    def on_epoch_end(self): #initialize
        self.idx = 0
        if self.is_train:
            self.shuffle_item()
        
    def get_label(self):
        if self.is_train:
            raise Exception('train mode cannot return label')
        else :
            labels = []
            for (image,mask,label,name) in self.data:
                labels.append([label])
            return np.array(labels)
        
        
# class DataGenerator():
#     def __init__(self, image_dir, masking_dir,batch_size, shape, is_train= True, copy=True):
#         self.image_dir = image_dir
#         self.masking_dir = masking_dir
#         self.batch_size = batch_size
#         self.is_train = is_train
#         self.shape = shape

#         self.i = 0

#         self.get_file_name(copy)
#         self.shuffle_item()
    
#     def __len__(self):
#         return int(self.total_length/self.batch_size)
    
#     def get_file_name(self,copy):
#         image_files = []
#         for (path, dir, files) in os.walk(self.image_dir):
#             if path ==self.image_dir: # 현재 디렉토리는 넘김
#                 continue
#             numcopy = 9 if (copy and path.split('\\')[-1] == "Glaucoma") else 1
#             for file in files*numcopy:
#                 image_files.append(os.path.join(path, file))
        
#         mask_files = []
#         labels = []
#         for f in image_files:
#             label = os.path.dirname(f).split('\\')[-1]
#             labels.append(LABEL[label])
#             m = 'mask_'+os.path.basename(f)
#             m = os.path.join(self.masking_dir,m)
#             mask_files.append(m)
            
#         self.files = [i for i in zip(image_files, mask_files, labels)]
#         self.total_length = len(self.files)

#     def shuffle_item(self):
#         random.shuffle(self.files)

#     def get_batch(self,i):
#         return self.files[i*self.batch_size : (i+1)*self.batch_size]
    
#     def argumentation(self,img,mask,is_train):
#         ## resize
#         img = resize_image(img,self.shape)
#         mask = resize_image(mask,self.shape)
#         if is_train:
#             ## random한 각도로 image를 회전
#             angle = random.randint(1,359)
#             img = image_rotate(img, angle)
#             mask = image_rotate(mask, angle)
        
#         ## get optic disk
#         img = crop_optic_disk(img,mask,self.shape)

#         ## image 정규화
#         img = normalize_img(img)
#         ## Adaptive_Histogram_Equalization
# #         img = Adaptive_Histogram_Equalization(img)
#         ## image shift
#         ## 50% 확률로 이미지 상하 혹은 좌우 반전
#         return img    

#     def get_item(self):
#         temp = self.get_batch(self.i)
#         self.i += 1

#         items = np.zeros((self.batch_size,)+self.shape)
#         y_true = np.zeros((self.batch_size,1))
#         for i,f in enumerate(temp):
#             image = image_loader(f[0])
#             mask = image_loader(f[1])
#             image = self.argumentation(image,mask,self.is_train)
            
#             items[i] = image
#             y_true[i] = f[2]
#         return items, y_true
            
#     def initialize(self):
#         self.i = 0
#         self.shuffle_item()
        