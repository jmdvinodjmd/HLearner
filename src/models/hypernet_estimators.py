'''
@author:
    Vinod Kumar Chauhan
    Institute of Biomedical Engineering
    University of Oxford, UK
'''
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from src.models.hn_utils import MLPFunctional

from copy import deepcopy

from src.models.hypernetworks import *
from src import utils
from src.training_loop import train

from sklearn.model_selection import train_test_split
from sklearn.model_selection import StratifiedKFold


class HLearner():
    '''
    '''
    def __init__(self, target_layers, activations, args, input_size, device):
        super(HLearner, self).__init__()
        self.args = args
        self.device = device
        self.net = HyperLearner(args.hypernet1, args, input_size, activations,
                    target_layers, args.emb_dim1, args.hn_drop_rate1, args.spect_norm1).to(device)

    def fit(self, X, y, t, logger, wandb, device):
        print(X.shape, t.shape, y.shape)
        # X_train, X_val, y_train, y_val, t_train, t_val = train_test_split(X, y, t, test_size=self.args.val_size, random_state=42, stratify=y[:,0] + y[:,0]*t[:,0])
        X_train, X_val, y_train, y_val, t_train, t_val = train_test_split(X, y, t, test_size=self.args.val_size, random_state=42, stratify=t[:,-1])

        self.net, results_dict = train(self.net, X_train, X_val, y_train, y_val, t_train, t_val, self.args, 
                         logger, wandb, name='', f_loss='mse')
       
        return results_dict

    def predict(self, X, t, device):
        self.net.eval()
        X = torch.tensor(X).to(device).float()
        t = torch.tensor(t).to(device).float()

        out = self.net(X, t)

        return out
        