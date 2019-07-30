import sys
import numpy as np
import cv2
from config import *
import skimage
from skimage.exposure import equalize_adapthist, adjust_gamma
from skimage.transform import rescale, resize, downscale_local_mean, rotate
import random

import keras 
from tensorflow.python.keras.utils.data_utils import Sequence

def image_loader(f):
    return skimage.io.imread(f)

def resize_image(img,shape):
    # cv2.INTER_NEAREST -- 이웃 보간법
    # cv2.INTER_LINEAR -- 쌍 선형 보간법
    # cv2.INTER_LINEAR_EXACT -- 비트 쌍 선형 보간법
    # cv2.INTER_CUBIC -- 바이큐빅 보간법
    # cv2.INTER_AREA -- 영역 보간법
    # cv2.INTER_LANCZOS4 -- Lanczos 보간법
    # 기본적으로 쌍 선형 보간법이 가장 많이 사용됩니다.
    # 이미지를 확대하는 경우, 바이큐빅 보간법이나 쌍 선형 보간법을 가장 많이 사용합니다.
    # 이미지를 축소하는 경우, 영역 보간법을 가장 많이 사용합니다.
    # 영역 보간법에서 이미지를 확대하는 경우, 이웃 보간법과 비슷한 결과를 반환합니다.
    # 출처 : https://076923.github.io/posts/Python-opencv-8/
    # cv2.resize(img, dsize=shape, interpolation=cv2.INTER_AREA)
    img = resize(img,shape)
    return img

def image_rotate(img, angle):
    return rotate(img, angle)

def random_gamma(img):
    gamma = random.uniform(0.5,2.0)
    gain = random.uniform(0.5,2.0)
    return adjust_gamma(img,gamma, gain)

def normalize_img(img):
    shape = img.shape
    img = np.float64(img.reshape(-1))
    img -= img.mean()
    img /= img.std()
    img = img.reshape(shape)
#     img = img/ 255
    return img

def crop_optic_disk(img,mk, margin = 3):
    img_shape = img.shape
    h = np.where(mk>0)[0]
    h = int(mk.shape[0]/2) if h.size == 0 else h
        
    w = np.where(mk>0)[1]
    w = int(mk.shape[1]/2) if w.size == 0 else w
    
    maxh = min(np.max(h)+margin, mk.shape[0])
    minh = max(np.min(h)-margin, 0)
    maxw = min(np.max(w)+margin, mk.shape[1])
    minw = max(np.min(w)-margin, 0)
    
    img = img[minh:maxh,minw:maxw,:]
    img = resize(img, img_shape)
    return img

def Adaptive_Histogram_Equalization(img):
#     clahe = cv2.createCLAHE(clipLimit=1, tileGridSize=(8,8))
#     for i in range(3):
#         ch = img[:,:,i]
#         ch = clahe.apply(ch)
#         img[:,:,i] = ch
    return equalize_adapthist(img)         
        
def print_progress(total,i):
    dot_num = int(i/total*100)
    dot = '>'*dot_num
    empty = '_'*(100-dot_num)
    sys.stdout.write(f'\r [{dot}{empty}] {i} Done')
    if i == total:
        sys.stdout.write('\n')

class DataGenerator(keras.utils.Sequence):
    def __init__(self, image_dir, masking_dir,batch_size, load_shape, disc_shape, is_train= True, copy=True,sample =None):
        self.image_dir = image_dir
        self.masking_dir = masking_dir
        self.batch_size = batch_size
        self.is_train = is_train
        self.load_shape = load_shape
        self.disc_shape = disc_shape
        if (sample is not None) & (not isinstance(sample,int)):
            raise ValueError("sample is not integer")
        else:
            self.sample = sample
        self.idx = 0

        self.load_data(copy)
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
            
            mask = mask[:,:,0]
            mask = np.where(mask<=100,0.0,mask)
            mask = np.where(mask>100,1.0,mask)
            mask = mask[:,:,np.newaxis]
            #mask = resize_image(mask,self.load_shape)
            #image = self.argumentation(image,mask,self.is_train)
            data.append((image,mask,label,name))
        return data
    
    def argumentation(self,img,mask,is_train):
        
        if is_train:
            ## random한 각도로 image를 회전
            angle = random.randint(1,35)*10
            img = image_rotate(img, angle)
            mask = image_rotate(mask, angle)
        
        ## get optic disk
        img = crop_optic_disk(img,mask)
        ## Adaptive_Histogram_Equalization
        img = Adaptive_Histogram_Equalization(img)
        
        if is_train:
            ## random gamma
            img = random_gamma(img)
        
        ## image 정규화
        img = normalize_img(img)
        ## image shift
        ## 50% 확률로 이미지 상하 혹은 좌우 반전
        img = resize_image(img,self.disc_shape)
        return img    

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
            image = self.argumentation(image,mask,self.is_train)
            x_train[i] = image
            y_true[i] = label
            names.append(name)

        return x_train, y_true
            
    def on_epoch_end(self): #initialize
        self.idx = 0
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
        