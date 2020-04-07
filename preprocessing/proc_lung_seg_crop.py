from skimage.io import imread, imsave
import os
from skimage.measure import regionprops
import skimage.morphology as skm
from skimage.transform import resize
from skimage.color import rgb2gray
import numpy as np
from skimage.measure import label   
from skimage.morphology import remove_small_objects
from scipy.ndimage.morphology import binary_closing, binary_dilation, binary_fill_holes
import glob
import pandas as pd
from multiprocessing import Pool

def getLargestCC(segmentation):
    labels = label(segmentation)
    assert( labels.max() != 0 ) # assume at least 1 CC
    largestCC = labels == (np.argmax(np.bincount(labels.flat)[1:]) + 1)
    return largestCC

def maybe_create(path):
    if not os.path.exists(path):
        os.mkdir(path)
    return

def clean_seg(seg):
    labels = label(seg)
    labels = remove_small_objects(labels)
    seg = labels > 0.5
    seg = binary_closing(seg, structure = np.ones((10, 10)))
    seg = binary_fill_holes(seg)
    return seg


def get_masks(img):
    if (img.shape[0] > 512) or (img.shape[1] > 512):
        img = resize(img, (512, 512), preserve_range=True, anti_aliasing=True)
        img = np.clip(img, 0, 255)
        img = np.uint8(img)

    seg = getLargestCC(img[:, :, 2] > 175)  

    labels = label(~np.pad(seg, [[1, 1], [1, 1]], constant_values = False))

    body_mask = clean_seg((labels != labels[0, 0])[1:-1, 1:-1])
    lung_mask = clean_seg(((labels != labels[0, 0]) & (labels > 0))[1:-1, 1:-1])
    return img, body_mask, lung_mask        
        
        
def bbox2(img):
    ## https://stackoverflow.com/questions/31400769/bounding-box-of-numpy-array
    
    rows = np.any(img, axis=1)
    cols = np.any(img, axis=0)
    rmin, rmax = np.where(rows)[0][[0, -1]]
    cmin, cmax = np.where(cols)[0][[0, -1]]

    return rmin, cmin, rmax, cmax

def create_img_save(img, body_mask, lung_mask):
    img_save = np.zeros(img.shape).astype('uint8')
    img_save[:, :, 0] = np.uint8(rgb2gray(img) * 255) * body_mask
    img_save[:, :, 1] = body_mask.astype('uint8')
    img_save[:, :, 2] = lung_mask.astype('uint8')
    return img_save


def proc(var_in):
    c, p = var_in
    fn_prob = os.path.join('TB_prob/probs', c, p + '.csv')
    df = pd.read_csv(fn_prob)
    df = df.sort_values(by = 'prob_c0', axis = 0)
    
    ir = 0
    
    path_save = os.path.join(path0, c, p)
    if os.path.exists(path_save):
        return
    
    for f in df.filename:
        if ir > 9:
            break
            
        fn = os.path.join('/data', c, p, f)
        if not os.path.exists(fn):
            continue
            
        f = '.'.join(f.split('.')[:-1]) 

        try: 
            fn_save = os.path.join(path0, c, p)
            maybe_create(fn_save)
            fn_save = os.path.join(path_save, ('%02d_' % ir) + f + '.png')

            if os.path.exists(fn_save):
                continue    

            img = imread(fn)    
            img, body_mask, lung_mask = get_masks(img)
            
            if lung_mask.sum() / body_mask.sum() < 0.2:
                continue 
                
            ir += 1        
            img_save = create_img_save(img, body_mask, lung_mask)

            i0, j0, i1, j1 = regionprops(img_save[:, :, 1])[0].bbox
            i0, j0 = max(i0 - 20, 0), max(j0 - 20, 0)
            i1, j1 = i1 + 20, j1 + 20
            img_save = img_save[i0:i1, j0:j1]

            imsave(fn_save, img_save[:, :, 0])

            fn_save =  os.path.join(path0 + '_with_mask', c, p)
            maybe_create(fn_save)
            fn_save = os.path.join(fn_save, ('%02d_' % ir) + f + '.png')

            imsave(fn_save, img_save)    

        except:
            print(fn + ' failed')    
    
if __name__ == "__main__":
    _path_data = 'Z://'
    path0 = 'lung_seg_crop'
    maybe_create(path0)
    maybe_create(path0 + '_with_mask')
    
    for c in os.listdir('TB_prob/probs'):
        maybe_create(os.path.join(path0, c))
        maybe_create(os.path.join(path0 + '_with_mask', c))
    
    info = [f.split('/')[-2:] for f in glob.glob('TB_prob/probs/*/*.csv')]
    info = [(i[0], i[1][:-4]) for i in info]
    pool = Pool(10)
    pool.map(proc, info)
        