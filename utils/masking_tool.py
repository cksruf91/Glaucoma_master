#-*- coding:utf-8 -*-
import cv2
import numpy as np 
import sys
import os
import argparse
import random
sys.path.append("..")
from config import *
from image_util import image_rotate, resize_image

def args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--im', type=str, required=True
                        , help='image file')  # number of class
    parser.add_argument('--m', type=int, required=True, choices=[0,1]
                        , help='modify exist mask image')  # number of class
    args = parser.parse_args()
    return args

# def resize_image(img,shape):
#     img = cv2.resize(img, dsize=shape, interpolation=cv2.INTER_AREA) 
#     return img

def cleaner(mak, ignore =1):
    idx = []
    tmp = np.pad(mak,3,mode = 'constant', constant_values=0)[:,:,3:6]
    for i in range(tmp.shape[0]-3):
        for j in range(tmp.shape[1]-3):
            if np.sum(tmp[i:i+3,j:j+3,:]> 100) <= ignore*3 :
                idx.append((i,j,3))
    print(len(idx))
    for (i,j,_) in idx:
        tmp[i:i+3,j:j+3,:] = np.zeros((3,3,3))
    return tmp[3:-3,3:-3,:]

# Image load 함수
def image_mask_load(f,image_size, m):
    img = cv2.imread(f, cv2.IMREAD_COLOR)
    img = resize_image(img, image_size, True).astype(int)
    img = img.astype(np.uint8)
    if m == 1:
        mask_im = 'mask_'+os.path.basename(f)
        mask = cv2.imread(os.path.join(MASK_LOC, mask_im), cv2.IMREAD_COLOR)
        mask = resize_image(mask, image_size, True)
        mask = mask.astype(np.uint8)
        
    else :
        mask = np.zeros(image_size).astype(np.uint8)
    return img, mask

# Mouse Callback함수
def draw_circle(event, x,y, flags, param):
    global ix,iy, drawing, mode, IMAGE_SHAPE, remove
    
    #마우스를 누른 상태
    if event == cv2.EVENT_LBUTTONDOWN:
        drawing = True
        if remove == True:
            cv2.circle(mask,(x,y),5,[0,0,0],-1)
        else:
            cv2.circle(mask,(x,y),5,[255,255,255],-1)

    elif event == cv2.EVENT_MOUSEMOVE: # 마우스 이동
        if drawing == True:            # 마우스를 누른 상태 일경우
            if remove:
                cv2.circle(mask,(x,y),5,[0,0,0],-1)
            else:
                cv2.circle(mask,(x,y),5,[255,255,255],-1)


    elif event == cv2.EVENT_LBUTTONUP:
        drawing = False;             # 마우스를 때면 상태 변경
        if remove:
            cv2.circle(mask,(x,y),5,[0,0,0],-1)
        else:
            cv2.circle(mask,(x,y),5,[255,255,255],-1)

            
# def main():
arg = args()    

name = arg.im
if 'mask_' in name:
    name = name.replace('mask_','')

folder = IMAGE_LOC
for path, dirs, file in os.walk(folder):
    for f in file:
        if f == name:
            load = os.path.join(path,name)

print(load)
if not os.path.isfile(load):
    raise ValueError('not exist file')


mask_save_location = os.path.join(MASK_LOC,'GT')

drawing = False #Mouse가 클릭된 상태 확인용
remove = False
mode = True # True이면 사각형, false면 원
ix,iy = -1,-1

IMAGE_SHAPE = (512,512,3) #(256,256,3) # 512,512,3
print(IMAGE_SHAPE)
img, mask = image_mask_load(load,IMAGE_SHAPE,arg.m)

cv2.namedWindow('image')
cv2.setMouseCallback('image',draw_circle)

while True:
    alpha=50
    dst = cv2.addWeighted(img,0.5, mask,0.5 , 0.0)
    cv2.imshow('image', dst)
    cv2.imshow('mask', mask)

    k = cv2.waitKey(1) & 0xFF

    if k == ord('r'):    # 이미지 초기화
        # mode = not mode
        img, mask = image_mask_load(load,IMAGE_SHAPE,arg.m)
    elif k == ord('s'): # s를 누르면 저장
        maskname = os.path.join(mask_save_location,'mask_'+name)
        print(cv2.imwrite(maskname,mask))
    elif k == ord('m'): # m을 누르면 지우기 모드
        remove = not remove
    elif k == ord('o'): # o이미지 회전
        angle = random.randint(1,35)*10
        img = image_rotate(img,angle)
        mask = image_rotate(mask,angle)
        mask = np.where(mask<=0.3,0.0,mask)
        mask = np.where(mask>0.3,1.0,mask)

    elif k == 27:        # esc를 누르면 종료
        break

cv2.destroyAllWindows()

    
# if __name__ == "__main__":
#     main()