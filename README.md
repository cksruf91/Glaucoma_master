Glaucoma Master
- - - 

Glaucoma Diagnosis model

## meta
- Python 3.6.8
- conda 4.6.14
- tensorflow-gpu 1.13.1
- CUDA 9.0
- NVIDIA cuDNN v7.5.0 (Feb 21, 2019), for CUDA 9.0
- opencv 3.4.3



## Train segmentation
train segmentation model 

```python
python train_seg.py
```
you can test source code with -t option 
(it will load small amount of data for testing)
```python
python train_seg.py -t
```

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
```python
python binary_dump.py
python binary_dump.py -t # test mode
```


## Train model
train Diagnosis model 

```python
python train.py
```


## Result 
you can verify model performance in jupyter notebook
> result.ipynb