import os
import sys
import subprocess
import argparse

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--gpu', help='Decide which GPU. Default 0', default="0")
    args = parser.parse_args()

    
    os.system('python train.py --gpu %s --cv_fold 0' % str(args.gpu))
    os.system('python train.py --gpu %s --cv_fold 1' % str(args.gpu))
    os.system('python train.py --gpu %s --cv_fold 2' % str(args.gpu))
    os.system('python train.py --gpu %s --cv_fold 3' % str(args.gpu))
    os.system('python train.py --gpu %s --cv_fold 4' % str(args.gpu))
        
