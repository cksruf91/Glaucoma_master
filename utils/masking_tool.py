#-*- coding:utf-8 -*-
import cv2
import numpy as np 
import sys
import os
import argparse
sys.path.append("..")
from config import *

def args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--im', type=str, required=True
                        , help='image file')  # number of class
    parser.add_argument('--m', type=int, required=True, choices=[0,1]
                        , help='modify exist mask image')  # number of class
    args = parser.parse_args()
    return args
arg = args()

def resize_image(img,shape):
    img = cv2.resize(img, dsize=shape, interpolation=cv2.INTER_AREA) 
    return img



masking_image = 'D:\\Projects\\my_Glaucoma_project_v2\\dataset\\Glaucoma\\mask_od'

name = arg.im
if 'mask_' in name:
    name = name.replace('mask_','')

if 'g' in name.split('.')[0] or 'n' in name.split('.')[0]:
    folder = TRAIN_IMAGE
    for path,dit,file in os.walk(folder):
        for f in file:
            if f == name:
                load = os.path.join(path,name)
                
elif 'V' in name.split('.')[0]:
    folder = TEST_IMAGE
    for path,dit,file in os.walk(folder):
        for f in file:
            if f == name:
                load = os.path.join(path,name)
else :
    raise ValueError('Unknown file')

print(load)
if not os.path.isfile(load):
    raise ValueError('not exist file')


mask_save_location = os.path.join(masking_image,'GT')

drawing = False #Mouse가 클릭된 상태 확인용
remove = False
mode = True # True이면 사각형, false면 원
ix,iy = -1,-1

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
    


# Image load 함수
def image_mask_load(f,image_size, m):
    img = cv2.imread(f, cv2.IMREAD_COLOR)
    img = cv2.resize(img, dsize=image_size[:2], interpolation=cv2.INTER_AREA)
    if m == 1:
        mask_im = 'mask_'+os.path.basename(f)
        mask = cv2.imread(os.path.join(masking_image, mask_im), cv2.IMREAD_COLOR)
        mask = resize_image(mask,IMAGE_SHAPE[:2])
        mask = mask.astype(np.uint8)
        
    else :
        mask = np.zeros(image_size).astype(np.uint8)
    return img, mask

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

    elif k == 27:        # esc를 누르면 종료
        break

cv2.destroyAllWindows()
