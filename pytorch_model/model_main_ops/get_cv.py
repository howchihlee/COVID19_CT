import os
import glob
import pandas as pd

path_to_cv_split = 'Z://cv.xlsx'
path0 = os.path.abspath('../../')
path_to_image_folder = os.path.join(path0, "body_seg_crop_with_mask\\")

def get_split(fold):
    ## cv: 0-4
    fn_all =  glob.glob(path_to_image_folder + "/*/*/*.png")
    fn_info = [(f.split('\\')[-2], f, int('Positive' in f)) for f in fn_all] ## (pid, fn, label)
    
    all_pid = set([f[0] for f in fn_info])

    df_cv = pd.read_excel(path_to_cv_split)
    print('read fold: ' + str(fold + 1))
    
    cv_key = 'cv%d' % (fold + 1)
    test_pid = set(df_cv[cv_key])
    
    not_test_pid = set([f for f in all_pid if f not in test_pid])
    val_pid = set([f for i, f in enumerate(sorted(not_test_pid)) if i % 10 == 1 ])
    train_pid = set([f for i, f in enumerate(sorted(not_test_pid)) if i % 10 != 1 ])
    
    train_info = [f for f in fn_info if f[0] in train_pid]
    test_info = [f for f in fn_info if f[0] in test_pid]
    val_info = [f for f in fn_info if f[0] in val_pid]
    
    assert (len(set(train_pid) & set(test_pid)) == 0)
    assert(len(set(train_pid) & set(val_pid)) == 0)
    assert(len(set(val_pid) & set(test_pid)) == 0)
    for i in range(2):
        assert(len(set([f[i] for f in train_info]) & set([f[i] for f in test_info])) == 0)
        assert(len(set([f[i] for f in train_info]) & set([f[i] for f in val_info])) == 0)
        assert(len(set([f[i] for f in val_info]) & set([f[i] for f in test_info])) == 0)
        
    assert((len(train_info) +len(val_info) + len(test_info)) == len(fn_info))
    assert((len(train_pid) + len(val_pid) + len(test_pid)) == len(all_pid))

    return train_info, val_info, test_info