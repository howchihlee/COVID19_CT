import os
import glob
import pandas as pd

path_to_clinical_info = '/data/COVID19_clinical_data/clinical_info_joint.csv'
path0 = os.path.abspath('../../')
path_to_image_folder = os.path.join(path0, "lung_seg_crop_with_mask")

def _read_clinical_info():
    df = pd.read_csv(path_to_clinical_info)
    feature_list = ['Sex', 'Age', 'Exposure',
           'Fever', 'Sputum', 'Cough', 'Temperature',
           'WhiteBloodCell', 'Neutrophil', 'NeutrophilPercent', 'Lymphocyte',
           'LymphocytePercent']

    pid2features = {}
    for p, v in zip(df.pid, df[feature_list].values):
        pid2features[p] = v
    
    return pid2features

def _read_all_info(split_key):
    df_cv = pd.read_csv(path_to_clinical_info)
    pid2label = {p:l for p, l in zip(df_cv.pid, df_cv.label)}
    
    fn_all =  glob.glob(path_to_image_folder + "/*/*/*.png")
    fn_info = [(f.split('/')[-2], f, pid2label[f.split('/')[-2]]) for f in fn_all if f.split('/')[-2] in pid2label] ## (pid, fn, label)
    pid2features = _read_clinical_info()
    
    split_pid = set(df_cv.pid[df_cv.set ==split_key])
    
    ## subset samples with clinical data
    split_pid = set([p for p in split_pid if p in pid2features])
    
    fn_info = [f for f in fn_info if f[0] in split_pid]
    
    return fn_info, split_pid, pid2features
    
def get_train_split(fold):
    ## cv: 0-4
    print('read fold: ' + str(fold + 1))
    fn_info, train_pid, pid2features = _read_all_info('train')
    
    df = pd.read_csv('../ct_lung_maxpool_res18/model_cv%d/train_info.csv' % fold)
    train_info = [v for v in set(zip(df['0'], df['1'], df['2'])) if v[0] in pid2features]
    
    df = pd.read_csv('../ct_lung_maxpool_res18/model_cv%d/cv_info.csv' % fold)
    cv_info = [v for v in set(zip(df['0'], df['1'], df['2'])) if v[0] in pid2features]
   
    for i in range(2):
        assert(len(set([f[i] for f in train_info]) & set([f[i] for f in cv_info])) == 0)

    return train_info, cv_info, pid2features

def get_eval_split():
    return _read_all_info('val')

def get_test_split():
    return _read_all_info('test')