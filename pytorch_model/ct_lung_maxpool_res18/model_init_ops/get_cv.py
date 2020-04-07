import os
import glob
import pandas as pd

path_to_cv_split = '/data/COVID19_clinical_data/clinical_info_joint.csv'
path0 = os.path.abspath('../../')
path_to_image_folder = os.path.join(path0, "lung_patch")

def _read_all_info(split_key):
    df_cv = pd.read_csv(path_to_cv_split)
    pid2label = {p:l for p, l in zip(df_cv.pid, df_cv.label)}
    
    fn_all =  glob.glob(path_to_image_folder + "/*/*/*.png")
    fn_info = [(f.split('/')[-2], f, pid2label[f.split('/')[-2]]) for f in fn_all if f.split('/')[-2] in pid2label] ## (pid, fn, label)
    
    split_pid = set(df_cv.pid[df_cv.set ==split_key])
    fn_info = [f for f in fn_info if f[0] in split_pid]
    return fn_info, split_pid
    
def get_train_split(fold):
    ## cv: 0-4
    print('read fold: ' + str(fold + 1))
    fn_info, train_pid = _read_all_info('train')
    
    cv_pid = set([f for i, f in enumerate(sorted(train_pid)) if i % 5 == fold ])
    train_pid = set([f for i, f in enumerate(sorted(train_pid)) if i % 5 != fold ])

    train_info = [f for f in fn_info if f[0] in train_pid]
    cv_info = [f for f in fn_info if f[0] in cv_pid]
    
    assert (len(set(train_pid) & set(cv_pid)) == 0)
    
    for i in range(2):
        assert(len(set([f[i] for f in train_info]) & set([f[i] for f in cv_info])) == 0)

    return train_info, cv_info


def get_eval_split():
    return _read_all_info('val')[0]

def get_test_split():
    return _read_all_info('test')[0]