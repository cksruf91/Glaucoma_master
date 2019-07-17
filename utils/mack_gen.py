# -*- coding:utf-8 -*-
from __future__ import print_function
import os
import sys

import cv2
import numpy as np

from config import PROJECT

if len(sys.argv) < 2:
    raise ValueError('Name required')
name = sys.argv[1]
if 'g' in name.split('.')[0]:
    folder = os.path.join(PROJECT, 'dataset', 'Glaucoma', 'Training400', 'Glaucoma')
elif 'n' in name.split('.')[0]:
    folder = os.path.join(PROJECT, 'dataset', 'Glaucoma', 'Training400', 'Non-Glaucoma')
else:
    raise ValueError('Unknown file')

file = os.path.join(folder, name)
if not os.path.isfile(file):
    print(file)
    raise ValueError('not exist file')

mask_save_location = os.path.join(PROJECT, 'dataset', 'Glaucoma', 'mask_od', 'GT')
drawing = False  # Mouse가 클릭된 상태 확인용
mode = True  # True이면 사각형, false면 원
ix, iy = -1, -1


# Mouse Callback함수
def draw_circle(event, x, y, flags, param):
    global ix, iy, drawing, mode, IMAGE_SHAPE

    if event == cv2.EVENT_LBUTTONDOWN:  # 마우스를 누른 상태
        drawing = True
        # ix, iy = x,y
        # 127
        cv2.circle(img, (x, y), 5, [255, 255, 255], -1)
        cv2.circle(mask, (x, y), 5, [255, 255, 255], -1)

    elif event == cv2.EVENT_MOUSEMOVE:  # 마우스 이동
        if drawing is True:  # 마우스를 누른 상태 일경우
            cv2.circle(img, (x, y), 5, [255, 255, 255], -1)
            cv2.circle(mask, (x, y), 5, [255, 255, 255], -1)

    elif event == cv2.EVENT_LBUTTONUP:
        drawing = False  # 마우스를 때면 상태 변경
        # img[y,x] = [255,255,255]
        cv2.circle(img, (x, y), 5, [255, 255, 255], -1)
        cv2.circle(mask, (x, y), 5, [255, 255, 255], -1)


# Image load 함수
def image_load(location, image_size):
    image = cv2.imread(location, cv2.IMREAD_COLOR)
    image = cv2.resize(image, dsize=image_size, interpolation=cv2.INTER_AREA)
    return image


IMAGE_SHAPE = (499, 499, 3)

img = image_load(file, IMAGE_SHAPE[:2])
mask = np.zeros(IMAGE_SHAPE)

cv2.namedWindow('image')
cv2.setMouseCallback('image', draw_circle)

while True:
    cv2.imshow('image', img)
    cv2.imshow('mask', mask)

    k = cv2.waitKey(1) & 0xFF

    if k == ord('r'):  # 이미지 초기화
        # mode = not mode
        img = image_load(file, IMAGE_SHAPE[:2])
        mask = np.zeros(IMAGE_SHAPE)
    elif k == ord('s'):  # esc를 누르면 종료
        maskname = os.path.join(mask_save_location, 'mask_' + name)
        print(cv2.imwrite(maskname, mask))

    elif k == 27:  # esc를 누르면 종료
        break

cv2.destroyAllWindows()
