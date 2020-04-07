import argparse
import os
import glob
import numpy as np
import h5py

import torch
import torchvision.models as tmodels
import torch.nn as nn
import torch.nn.functional as F
import torch.utils.data as tdata

import pandas as pd
import logging

from model import construct_model
from helpers import polyak_update
from helpers import *


from model_main_ops import get_cv
from model_main_ops.datareader import DataReader, train_aug_ops, constant_ops
from model_main_ops.performance_metric import ScoreRecorder, compute_metric

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--cv_fold', help='cv fold', required=True)
    parser.add_argument('--gpu', help='Decide which GPU. Default 0', default="0")
    args = parser.parse_args()

    _gpu_id               = str(args.gpu)
    cv_fold               = int(args.cv_fold)
    
    os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"   # see issue #152
    os.environ["CUDA_VISIBLE_DEVICES"] = _gpu_id

    _folder_to_save = 'model_cv%d' % cv_fold
    maybe_create(_folder_to_save)
    fn_img_prob = os.path.join(_folder_to_save, 'prob_img.csv')
    fn_ave_prob = os.path.join(_folder_to_save, 'prob_ave.csv')
    fn_log = os.path.join(_folder_to_save, 'log.txt')

    opt_lr = 0.002
    opt_wdecay = 0.0001
    total_epoch = 40
    ## loss objective
    
    citerion_clf = torch.nn.BCEWithLogitsLoss()

    train_info, cv_info, pid2features = get_cv.get_train_split(cv_fold)
    
    pd.DataFrame(train_info).to_csv(os.path.join(_folder_to_save, 'train_info.csv'), index = False)
    pd.DataFrame(cv_info).to_csv(os.path.join(_folder_to_save, 'cv_info.csv'), index = False)
    
    print(len(train_info), len(cv_info))
    print('number of positive images %d, %d' % (sum([f[2] for f in train_info]), sum([f[2] for f in cv_info])))

    datareader = DataReader(train_info, pid2features, transform=train_aug_ops)
    train_generator = tdata.DataLoader(datareader, batch_size=16, shuffle=True, pin_memory = True)

    model = construct_model()
    model.load_partial_state_dict(torch.load('../ct_lung_maxpool_res18/model_cv%d/model_init_last.pth' % cv_fold))
    model.cuda()     

    optimizer = torch.optim.Adam(model.parameters(), lr=opt_lr, weight_decay=opt_wdecay)
    scheduler = torch.optim.lr_scheduler.ExponentialLR(optimizer, 0.95, last_epoch=-1)

    cv_datareader = DataReader(cv_info, pid2features, transform=constant_ops)
    cv_generator = tdata.DataLoader(cv_datareader, batch_size=16, shuffle=False, pin_memory = True)

    best_score = np.inf
    best_score_ema = np.inf

    delete_if_exist(fn_log)
    logger = setup_file_logger(fn_log)
    score_recorder = ScoreRecorder(logger)

    model_info = [[model, cv_generator, [(f[0], f[2]) for f in cv_info]],
                 ]
    
    total_step = 0.
    for epoch in range(total_epoch):  # loop over the dataset multiple times

        model.train()    

        for item_train in train_generator:
            optimizer.zero_grad()

            loss = model.train_update(item_train, citerion_clf)
            score_recorder.update_train_loss(loss.item())

            loss.backward()
            optimizer.step()

            total_step += 1

            if total_step % 100 == 0:
                tmp = score_recorder.output_score(model_info)      
                val_loss = tmp[0] ## need to match the order in model_info
                
                if (epoch > total_epoch // 2) and (best_score > val_loss):
                    torch.save(model.state_dict(), os.path.join(_folder_to_save, 'model_best.pth') )
                    best_score = val_loss

        scheduler.step()
        torch.save(model.state_dict(), os.path.join(_folder_to_save, 'model_last.pth'))
    logger.info('best_score: %.4f' % best_score)
        
        