import skimage
import numpy as np
import scipy
from skimage import transform as sktr
from skimage import morphology
import os
import albumentations as Alb
import torch
import torchvision.models as tmodels
from cv2 import BORDER_CONSTANT, BORDER_REPLICATE, BORDER_REFLECT
import torch.nn as nn
import torch.nn.functional as F
#import torchvision.transforms as ttransforms
import torch
import torchvision.models as tmodels
import torch.utils.data as tdata
from skimage.io import imread
import random

constant_ops = Alb.PadIfNeeded(512, 512, border_mode=BORDER_REFLECT)

train_aug_ops = Alb.Compose([
        Alb.HorizontalFlip(),
        Alb.GridDistortion(border_mode=BORDER_REFLECT),
        #Alb.ElasticTransform(approximate=True, border_mode=BORDER_CONSTANT),
        Alb.ShiftScaleRotate(shift_limit=0.1, scale_limit=0.1, rotate_limit=30, 
                             p=1.0, border_mode=BORDER_REFLECT),
        Alb.CoarseDropout(),
         ], p=1)


def apply_op(img, op):
    ## warper for apply an image operation op to 
    ## the numpy array img of size [channel, heignt, width]
    ## channel might be limited to 1 or 3, depending on the operation
    img = op(image=img)['image']
    return img

def proc_img(img, transform = None, random_mean = False):    
    if transform is not None:
        img = apply_op(img, transform)
    img = np.transpose(img, axes = [2, 0, 1]).astype('float32')  
    #img = img[:, np.newaxis, :, :]
    return img

class DataReader(tdata.Dataset):
    def __init__(self, data, transform = None):
        super(DataReader, self).__init__()
        self.data = data ## list of (pid, filename, label)
        self.transform = transform
        self.num_sample = len(self.data)
        #self.path = 'processed_data/image_patch/'
        
    def __len__(self):
        return self.num_sample
        
    def __getitem__(self, index):  
        
        pid, fn, label = self.data[index]

        img = imread(fn)
        
        img = proc_img(img[:, :, np.newaxis], transform = self.transform)
        
        return img, np.float32([label])
    