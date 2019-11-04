import numpy as np
import random
import warnings
import skimage
from skimage.exposure import equalize_adapthist, adjust_gamma
from skimage.transform import rescale, resize, downscale_local_mean, rotate
from skimage.morphology import opening, closing, square, disk
from skimage.util import invert
import polarTransform

warnings.filterwarnings(action='ignore') 

def image_loader(f):
    return skimage.io.imread(f)

def resize_image(img, shape, keeprange=False):
    return resize(img, shape, preserve_range=keeprange)

def image_rotate(img, angle):
    return rotate(img, angle, preserve_range=False)

def random_gamma(img):
    gamma = random.uniform(1.5,0.5)
    gain = random.uniform(1.5,0.5)
    return adjust_gamma(img, gamma, gain)

def random_invert_image(img):
    iv=random.choice([True, False])
    if iv:
        return invert(img)
    else:
        return img

def opening_image(img,size):
#     zero = np.zeros(img.shape)
#     for i in range(img.shape[-1]):
#         zero[:,:,i] = opening(img[:,:,i], disk(size))
    return opening(img, disk(size))

def closing_image(img, size):
#     zero = np.zeros(img.shape)
#     for i in range(img.shape[-1]):
#         zero[:,:,i] = closing(img[:,:,i], disk(size))
    return closing(img, disk(size))
    
    
def random_crop(img, mask=None, prop=0.2):
    ishape = img.shape
    if mask is not None:
        mshape = mask.shape
        if ishape[:2] != mshape[:2]:
            raise ValueError(f'image and maks is not same {ishape}, {mshape}')
    h = int(ishape[0]*prop)
    h_crop1 = random.choice(range(h))
    h_crop2 = h - h_crop1
    v = int(ishape[1]*prop)
    v_crop1 = random.choice(range(v))
    v_crop2 = v - v_crop1
    
    img = resize_image(img[h_crop1:-h_crop2,v_crop1:-v_crop2,:], ishape)
    if mask is not None:
        mask = resize_image(mask[h_crop1:-h_crop2,v_crop1:-v_crop2,:], mshape)
        return img, mask
    else :
        return img

def Adaptive_Histogram_Equalization(img,cl=0.03):
    for i in range(3):
        ch = img[:,:,i]
        transformed = equalize_adapthist(ch,clip_limit=cl)
        img[:,:,i] = transformed
    return img

def random_flip_image(img, horizon=True,vertical=True):
    if img.ndim != 3:
        raise ValueError(f'exception flip_image: expected dim 3 but got {train_images[0].ndim}')
    if horizon:
        if random.choice([True, False]):
            img = np.flip(img,0)
    if vertical:
        if random.choice([True, False]):
            img = np.flip(img,1)
    return img

def normalize_img(img):
    shape = img.shape
    img = np.float64(img.reshape(-1))
    img -= img.mean()
    img /= img.std()
    img = img.reshape(shape)
    return img

def per_chenel_normalize(img):
    normed = np.zeros(img.shape)
    for i in range(img.shape[-1]):
        temp = img[:,:,i]
        temp -= temp.mean()
        temp /= temp.std()
        normed[:,:,i] = temp
    return normed

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

def polartransform_image(img,angle):
    img , _ = polarTransform.convertToPolarImage(img, initialAngle=angle,
                                                 finalAngle=np.pi*2+angle, hasColor=True,border = 'nearest')
    img = img.transpose(1,0,2)
    return np.clip(img,0,1)

def polar(img, mask):
    
    for angle in range(0,360,10):
        test = polartransform_image(mask, angle)
        test = test.max(axis=0).max(-1) > 0.5
        trim = int(test.shape[0]*.3/2)
        test = np.concatenate([test[:trim],test[:trim]])
        if any(test)==False:
            break
    
    transfrom_im = polartransform_image(img,angle)
    transfrom_mk = polartransform_image(mask,angle)
    trim = int(transfrom_im.shape[0]/3)
    
    return transfrom_im[:-trim,:,:], transfrom_mk[:-trim,:,:]