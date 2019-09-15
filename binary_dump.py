import os
import h5py
import pickle
import numpy as np

from config import *
from utils.image_util import *
from utils.util import print_progress

    
def preprocess(files):    
    image, mask, label, name = files
    image = image_loader(image)
    mask = image_loader(mask)
    
    ## resize
    image = resize_image(image, (1634,1634,3))
    mask = resize_image(mask, (1634,1634,3))

    mask = mask[:,:,0]
    mask = np.where(mask<=0.3,0.0,mask)
    mask = np.where(mask>0.3,1.0,mask)
    mask = mask[:,:,np.newaxis]
    
    return image,mask,label,name

def to_hdf5(copy, infile_dir, masking_dir, outfile, sample):
    image_files = []
    file_name = []
    for (path, dir, files) in os.walk(infile_dir):
        if path == infile_dir: # 현재 디렉토리는 넘김
            continue
        ncopy = 9 if (copy and path.split('\\')[-1] == "Glaucoma") else 1
        for file in files * ncopy: # repeat 9 time
            image_files.append(os.path.join(path, file))
            file_name.append(file)

    mask_files = []
    labels = []

    for f in image_files:
        label = os.path.dirname(f).split('\\')[-1]
        labels.append(LABEL[label])
        m = 'mask_'+os.path.basename(f)
        m = os.path.join(masking_dir, m)
        mask_files.append(m)

    files = [i for i in zip(image_files, mask_files, labels, file_name)]
    random.shuffle(files)  
    total_length = len(files)
    
    if isinstance(sample, int):
        files = files[:sample]

    with h5py.File(outfile, 'a') as h5:
        for i,f in enumerate(files):
            
            image, mask, label, name = preprocess(f)
            print_progress(total_length, i+1)
            
            ## create space
            if i == 0:
                
                im = h5.create_dataset('image', (1,)+image.shape, chunks = True 
                                       ,maxshape = (None,)+ image.shape )
                mk = h5.create_dataset('mask', (1,)+mask.shape, chunks = True
                                       ,maxshape = (None,)+ mask.shape)
                la = h5.create_dataset('label', (1,), chunks = True
                                       ,maxshape = (None, ) )

            else:
                im.resize(im.shape[0] + 1 , axis = 0)
                mk.resize(mk.shape[0] + 1 , axis = 0)
                la.resize(la.shape[0] + 1 , axis = 0)

            ## save the data
            im[i:(i+1)] = image
            mk[i:(i+1)] = mask
            la[i:(i+1)] = label

if __name__ == "__main__":
    to_hdf5(True, TRAIN_IMAGE, MASK_LOC, TRAIN_DATASET, None )
    to_hdf5(False, TEST_IMAGE, MASK_LOC, TEST_DATASET, None )