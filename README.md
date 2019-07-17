# Glaucoma_master

## meta
- window 10 
- Python 3.6.8
- conda 4.6.14
- tensorflow-gpu 1.13.1
- CUDA 9.0
- NVIDIA cuDNN v7.5.0 (Feb 21, 2019), for CUDA 9.0

## Install
Ensure you a version of Python in the 2.7-3.6 range installed, then run:

    pip install -r requirements.txt

You will also need OpenCV.

## create train dataset

 - *PROJECT_LOCATION\\dataset\\~~*

```sh
python utils/tfrecord_util.py
```

## train model

```sh
python train.py
```
