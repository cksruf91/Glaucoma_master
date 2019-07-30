# Glaucoma Master

## meta
- Python 3.6.8
- conda 4.6.14
- tensorflow-gpu 1.13.1
- CUDA 9.0
- NVIDIA cuDNN v7.5.0 (Feb 21, 2019), for CUDA 9.0

## 학습용 image 저장 위치
 - *PROJECT_LOCATION\\dataset\\cifar10\\image\\*

## segmentation
효율적 학습을 위해 optic disk에 대한 masking image 생성

### masking image 생성

1. 마스킹 이미지 생성하기 위한 GUI tool
(GUI tool for create masking image)
```sh
python mack_gen.py --m 0 --im {image file name}
```
interface    
* r : image reload   
* m : change remove mode n drawing mode   
* s : save mask    
   - 위치 : c D:/Projects/my_Glaucoma_project/dataset/Glaucoma/mask_od/GT   

1. 생성되어 있는 masking 이미지 수정하기
```sh
python mack_gen.py --m 1 --im {image or masking file to modify}
```

1. Unet 학습용 tfrecode 생성
```sh
python seg_tfrecord_util.py
```

1. Unet 학습
```sh
python train_segmentation.py 
```

1. 학습된  Unet으로 masking 생성
```sh
python masking_generator.py
```


### model 학습용 tfrecode 파일 생성
```sh
python utils/tfrecord_util.py
```

## train model
```sh
python train.py
```