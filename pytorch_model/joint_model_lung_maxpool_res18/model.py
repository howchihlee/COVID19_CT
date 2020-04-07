import skimage
import numpy as np
import scipy
import os

import albumentations as Alb
import torch
import torchvision.models as tmodels
from cv2 import BORDER_CONSTANT, BORDER_REPLICATE, BORDER_REFLECT
import torch.nn as nn
import torch.nn.functional as F
#import torchvision.transforms as ttransforms
import torch.utils.data as tdata
from skimage.io import imread

def batch_to_cuda(item):
    return (i.cuda() for i in item)

class AddNormalNoise(torch.nn.Module):
    def __init__(self):
        super(AddNormalNoise, self).__init__()
        
    def forward(self, x):
        if self.training:
            x = x + torch.randn_like(x)
        return x
    
    
class MyModel(torch.nn.Module):
    def __init__(self, resnext):
        super(MyModel, self).__init__()
        self.backbone = resnext
        self.fc = nn.Conv2d(512, 1, kernel_size=(1, 1), stride=(1, 1), bias=True)
        self.fc_maxpool = nn.AdaptiveMaxPool2d((1, 1))
        self.fc_avgpool = nn.AdaptiveAvgPool2d((1, 1))
        self.dropout = nn.Dropout(p = 0.5)
        n_nodes = 64
        self.linear_net = nn.Sequential(torch.nn.BatchNorm1d(512 + 12),
                                        AddNormalNoise(),
                                        nn.Linear(512 + 12, n_nodes, bias=False), 
                                        nn.ReLU(),
                                        nn.BatchNorm1d(n_nodes),
                                        nn.Linear(n_nodes, n_nodes, bias=False),
                                        nn.ReLU(),
                                        nn.Linear(n_nodes, 1),
                                       )
        
        
        
    def pre_proc(self, x, add_noise = False):
        with torch.no_grad():
            if add_noise:
                x = x + torch.randn_like(x)
            x = x / 255.
        return x
    
    def forward(self, x0, x1, add_noise = False, is_pooling = True):
        ## x: [batch, 1, H, W]
        x = self.pre_proc(x0, add_noise = add_noise)
        
        x = self.backbone.conv1(x)
        x = self.backbone.bn1(x) 
        x = self.backbone.relu(x)
        x = self.backbone.maxpool(x)

        x = self.backbone.layer1(x)
        x = self.backbone.layer2(x)
        x = self.backbone.layer3(x)
        x = self.backbone.layer4(x)
        x = self.dropout(x)
        
        fea = torch.flatten(self.fc_avgpool(x), 1)
        fea = torch.cat([fea, x1], 1)

        p0 = self.linear_net(fea)
        
        p1 = self.fc(x)

        if is_pooling:
            p1 = self.fc_maxpool(p1)
            p1 = torch.flatten(p1, 1)
            
        
        return p0, p0, p1
    
    def load_partial_state_dict(self, state_dict):
        own_state = self.state_dict()
        for name, param in state_dict.items():
            if name not in own_state:
                 continue
            if isinstance(param, nn.Parameter):
                # backwards compatibility for serialized parameters
                param = param.data
            own_state[name].copy_(param)
    
    def train_update(self, train_tensor_tuple, criterion, get_accuracy = False):
        xb0, xb1, yb = batch_to_cuda(train_tensor_tuple) 
        ## xb: batch of images
        ## yb: labels
        outputs, _, outputs1 = self.forward(xb0, xb1, add_noise = True)
        loss = criterion(outputs, yb) + criterion(outputs1, yb)
        return loss
    
    def predict(self, item, is_prob = False):
        try:
            xb0, xb1, _ = item 
        except:
            xb0, xb1 = item ## without label
           
        xb0 = xb0.cuda()        
        xb1 = xb1.cuda()
        
        with torch.no_grad():
            outputs, _, _ = self.forward(xb0, xb1)
            if is_prob:
                outputs = torch.sigmoid(outputs)
        return outputs
    
    def get_logits(self, generator, is_prob = False):
        ## generator assumed to output xb (batch of data to predict), yb (batch of labels)
        self.eval()
        logits = []
        
        for item in generator:
            outputs = self.predict(item, is_prob)
            tmp = outputs.cpu().detach().numpy()
            logits.append(tmp)

        logits = np.vstack(logits)
        self.train()
        return logits
    
    
def construct_model():
    model_backbone = tmodels.resnet18(pretrained=True)
                                      
    model_backbone.fc = torch.nn.Linear(model_backbone.fc.in_features, 1)
    model_backbone.conv1 = nn.Conv2d(1, 64, kernel_size=(7, 7), stride=(2, 2), padding=(3, 3), bias=False)
    
    return MyModel(model_backbone)


