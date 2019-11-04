Glaucoma Master
- - - 
Glaucoma Diagnosis model

# 1. meta
- Python 3.6.8
- conda 4.6.14
- tensorflow-gpu 1.13.1
- CUDA 9.0
- NVIDIA cuDNN v7.5.0 (Feb 21, 2019), for CUDA 9.0
- opencv 3.4.3



## Train segmentation
train segmentation model 

```sh
python train_seg.py
```
you can test source code with -t option 
(it will load small amount of data for testing)

```sh
python train_seg.py -t
```

## test segmentation
visualize segmentation image for vaildation 
> test.ipynb

## Generate segmentation image 
generate segmentation image by using trained model 

```python
python mask_image_generator.py -m 1
python mask_image_generator.py -m 0
```
- options 
	- 1: train image   
	- 0: test image

## generate train/test dataset
it will create hdf5 dataset, dataset stored preprocessed image for speed up training

```sh
python binary_dump.py
python binary_dump.py -t # test mode
python binary_dump.py -c # copy mode
```
 - "copy mode" will copy Glaucoma image to balance Glaucoma and Non-Glaucoma image



## Train model
train Diagnosis model 

```sh
python train.py
```
## Result 
you can verify model performance in jupyter notebook
> result.ipynb