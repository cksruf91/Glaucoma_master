import sys
import numpy as np
import cv2
from config import *
import skimage
from skimage.exposure import equalize_adapthist, adjust_gamma
from skimage.transform import rescale, resize, downscale_local_mean, rotate
import random


def image_loader(f):
    return skimage.io.imread(f)

def resize_image(img,shape):
    img = resize(img,shape)
    #img = cv2.resize(img, dsize=shape, interpolation=cv2.INTER_AREA)
    return img

def Adaptive_Histogram_Equalization(img):
    #clahe = cv2.createCLAHE(clipLimit=1, tileGridSize=(8,8))
    #for i in range(3):
    #    ch = img[:,:,i]
    #    ch = clahe.apply(ch)
    #    img[:,:,i] = ch
    #img = img.astype(np.uint6)
    return equalize_adapthist(img)

def random_gamma(f):
    gamma = random.uniform(0.5,1.5)
    gain = random.uniform(0.5,1.5)
    return adjust_gamma(f,gamma, gain)

def print_progress(total,i):
    dot_num = int(i/total*100)
    dot = '>'*dot_num
    empty = '_'*(100-dot_num)
    sys.stdout.write(f'\r [{dot}{empty}] {i} Done')
    if i == total:
        sys.stdout.write('\n')

class DataGenerator():
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
        return img ,mask

    def shuffle_item(self):
        random.shuffle(self.data)

    def get_batch(self,i):
        from_ = i*self.batch_size
        to_ = (i+1)*self.batch_size
        if self.data_length < to_:
            raise IndexError("index out of range")
        return self.data[from_:to_]
    
    def get_item(self):
        temp = self.get_batch(self.idx)
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
            
    def initialize(self):
        self.idx = 0
        self.shuffle_item()
