from sklearn.metrics import roc_auc_score
import numpy as np
from sklearn.metrics import average_precision_score, roc_auc_score
from sklearn.metrics import log_loss
import pandas as pd

def compute_metric(pred_prob, labels, threshold = 0.3):
    pred = (pred_prob > threshold).astype('int')
    roc = roc_auc_score(labels, pred_prob)
    
    accuracy = (pred == labels).mean()

    FP = np.sum((labels == 0) & (pred == 1))
    FN = np.sum((labels == 1) & (pred == 0))
    TP = np.sum((labels == 1) & (pred == 1))
    TN = np.sum((labels == 0) & (pred == 0))

    sen = TP / (TP + FN + 1e-7)
    spec = TN / (TN + FP + 1e-7)
    PPV = TP / (TP + FP + 1e-7)
    NPV = TN / (TN + FN + 1e-7)
    return(accuracy, sen, spec, PPV, NPV, roc) ## 'acc, sen, spec, ppv, npv, auc_roc'

    
class ScoreRecorder():
    def __init__(self, logger):
        self.counter = 0
        self.logger = logger
        self._reset_train_loss()
        self.set_header()
        
    def set_header(self):
        msr = 'iter, train_loss, eval_loss, ' + ', '.join(['accuracy', 'sensitivity', 'specificity', 'ppv', 'npv', 'auc_roc']) 
        self.logger.info(msr)
        
    def _reset_train_loss(self):
        self.train_loss =  0.
        self.train_sample = 0.       
    
    def update_train_loss(self, current_loss):
        self.train_loss +=  current_loss
        self.train_sample += 1.  
        
    def output_score(self, model_info):
        scores = [self.counter, self.train_loss / (self.train_sample + 1)]
        scores_to_print, scores_to_return = self.get_scores(model_info)
        scores += scores_to_print
        scores = tuple(scores)
        
        msr = '%03d, ' + ('%.3f, ' * (len(scores) - 1)) 
        msr = msr % scores
        self.logger.info(msr)
        
        self.counter += 1
        self._reset_train_loss()
        return scores_to_return
        
    def get_scores(self, model_info):
        scores_to_print = []
        scores_to_return = []
        for m, generator, info in model_info:
            
            labels = np.array([i[1] for i in info])
            
            pred_prob = m.get_logits(generator, is_prob = True) 
            
            df = pd.DataFrame([f[0], f[1]] + [v[0]] for f, v in zip(info, pred_prob))
            df_mean = df.groupby([0, 1]).mean()
            per_patient_prob = df_mean[2].values
            per_patient_label = np.array([v[1] for v in df_mean.index])
            s0 = log_loss(labels, pred_prob, eps = 1e-7)
            scores_to_return += [s0]
            scores_to_print += [s0]
            scores_to_print += list(compute_metric(per_patient_prob, per_patient_label, threshold = 0.3))
            
        return scores_to_print, scores_to_return  