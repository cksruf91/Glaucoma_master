import numpy as np
import random
import warnings
import skimage
from skimage.exposure import equalize_adapthist, adjust_gamma
from skimage.transform import rescale, resize, downscale_local_mean, rotate
import polarTransform

warnings.filterwarnings(action='ignore') 

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
    return resize(img,shape)

def image_rotate(img, angle):
    return rotate(img, angle,preserve_range=False)

def random_gamma(img):
    gamma = random.uniform(1.5,1.0)
    gain = random.uniform(1.8,0.5)
    return adjust_gamma(img,gamma, gain)

def Adaptive_Histogram_Equalization(img,cl=0.03):
#     clahe = cv2.createCLAHE(clipLimit=1, tileGridSize=(8,8))
#     for i in range(3):
#         ch = img[:,:,i]
#         ch = clahe.apply(ch)
#         img[:,:,i] = ch
    return equalize_adapthist(img,clip_limit=cl)

def random_flip_image(img, horizon=True,vertical=True):
    if img.ndim != 3:
        raise ValueError(f'exception flip_image: expected dim 3 but got {train_images[0].ndim}')
    if horizon:
        img = np.flip(img,0)
    if vertical:
        img = np.flip(img,1)
    return img

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

def polartransform_image(img,angle):
    img , _ = polarTransform.convertToPolarImage(img, initialAngle=angle,
                                                 finalAngle=np.pi*2+angle, hasColor=True,border = 'nearest')
    img = img.transpose(1,0,2)
    return np.clip(img,0,1)