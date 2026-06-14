'''
    Traning and evaluation of the model.
'''
from cmath import exp
from math import gamma
import wandb
import os
from tqdm import tqdm
import sys
# import matplotlib.pyplot as plt
import time
# import datetime
# import argparse
import numpy as np
# import pandas as pd
from random import SystemRandom

import torch
import torch.nn as nn

import src.utils as utils
from src.data.load_dataset import get_data_loaders

from copy import deepcopy

from turtle import xcor
import torch

import numpy as np
from sklearn.model_selection import train_test_split
import sklearn
import pickle
import json
import numpy as np
from sklearn.preprocessing import StandardScaler, scale
from torch.utils.data import Dataset, DataLoader, random_split, WeightedRandomSampler

from src import utils


def train(model, X_train, X_val, y_train, y_val, t_train, t_val, args, logger, wandb, name, 
          f_loss, w_train=None, w_val=None):
    
    # best_model = deepcopy(model)
    best_model = model
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    experimentID = model.__class__.__name__ + '-' + args.dataset  #int(SystemRandom().random()*100000)
    # checkpoint
    ckpt_path = os.path.join('./results/checkpoints/', str(experimentID) + '.ckpt')
    logger.info("Experiment " + str(experimentID))

    logger.info('args:\n')
    logger.info(args)
    logger.info(sys.argv)

    ######################################################
    ############### Prepare model and data ###############
    train_loader, val_loader = get_loader_from(X_train, X_val, y_train, y_val, t_train, t_val, 
                                               device, args.batch_size, w_train=w_train, w_val=w_val)
    
    opt = torch.optim.SGD(model.parameters(), lr=0.005)
    # opt = torch.optim.Adam(model.parameters(), lr=args.lr1, weight_decay=args.weight_decay) #, args.lr1, args.weight_decay, weight_decay=1e-5 for l2-regularisation
    # scheduler = torch.optim.lr_scheduler.StepLR(opt, step_size=3, gamma=0.5, verbose=True)
    # scheduler = torch.optim.lr_scheduler.OneCycleLR(opt, max_lr=0.001, steps_per_epoch=len(train_loader), epochs=args.niters, verbose=True)
    # scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(opt, 'min', factor=0.5, patience=5, verbose=True)

    # print model architecture and track gradients using wandb
    logger.info(model)
    wandb.watch(model)

    ############### TRAINING LOOP ###############
    early_stopping = utils.EarlyStopping(patience=args.patience, path=ckpt_path, verbose=False, logger=logger)

    results_dict = {}
    results_dict['train loss'] = []
    results_dict['val loss'] = []
    for epoch in range(1, args.niters+1):
        model.train()
        train_loss = 0
        # train_bce = 0
        # train_mse = 0

        for data_list in train_loader:
            opt.zero_grad()

            # forward pass
            loss = get_loss(args, model, data_list, name, device, epoch)

            # backward pass
            loss.backward()

            if args.clip_val_tag:
                torch.nn.utils.clip_grad_value_(model.parameters(), clip_value=args.clip_value)
            elif args.clip_norm_tag:
                torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=args.clip_value)
                
            opt.step()
            train_loss += loss.item()
            # train_bce += (loss_bce).item()
            # train_mse += (loss_mse).item()
        
        model.eval()
        loss = test(args, device, 'Val', val_loader, model, name, logger, wandb, epoch)

        if epoch%50==0 or epoch==1:
            logger.info('Train Loss: {:.3f}'.format(train_loss/len(train_loader)))
            wandb.log({'Train Loss-'+name: train_loss/len(train_loader)})
            # logger.info('Train BCE:  {:.3f}'.format(train_bce/len(train_loader)))
            # logger.info('Train MSE:  {:.3f}'.format(train_mse/len(train_loader)))
            logger.info('Val Loss:   {:.3f}'.format(loss/len(val_loader)))
            wandb.log({'Val Loss-'+name: loss/len(val_loader)})
            # logger.info('Val BCE:    {:.3f}'.format((loss_bce)/len(val_loader)))
            # logger.info('Val MSE:    {:.3f}'.format((loss_mse)/len(val_loader)))
            logger.info('Epoch: {}\n'.format(epoch))
        
        # save results
        results_dict['train loss'].append(train_loss/len(train_loader))
        results_dict['val loss'].append(loss/len(val_loader))

        # scheduler.step(test_loss)
        early_stopping(loss/len(val_loader), model)
        if early_stopping.early_stop:
            logger.info("Early stopping.... Epochs:"+str(epoch) +" Val loss:" + str(loss/len(val_loader)))
            break

    # load the best model from early stopping
    best_model.load_state_dict(torch.load(ckpt_path))
    best_model.eval()
    loss = test(args, device, 'Val', val_loader, best_model, name, logger, wandb, epoch)
    results_dict['Best Val Loss'] = loss

    return best_model, results_dict

def test(args, device, setting, test_loader, model, name, logger, wandb, epoch):
    # loss_bce = 0
    loss_mse = 0
    
    # with torch.no_grad():
    for data_list in test_loader:
        loss = get_loss(args, model, data_list, name, device, epoch)        
        loss_mse += loss.item()
    
    return loss_mse

def get_loss(args, model, data_list, name, device, epoch):
    X, y, t = data_list[0], data_list[1], data_list[2]

    if args.model == 'HLearner' or args.model == 'SLearner' or args.model == 'xSLearner':
        out = model(X, t)
        
        loss = 0
        for i in range(out.shape[1]):
            if args.tasks[i]=='cont':
                mask = torch.isnan(y[:,i].squeeze())
                loss += nn.MSELoss()(out[:,i].squeeze()[~mask], y[:,i].squeeze()[~mask])
            else:
                mask = torch.isnan(y[:,i].squeeze())
                loss += nn.BCELoss()(out[:,i].squeeze()[~mask], y[:,i].squeeze()[~mask])
        
    # elif args.model == 'SLearner':
    #     out = model(X, t)
    #     loss = 0
    #     for i in range(out.shape[1]):
    #         mask = torch.isnan(y[:,i].squeeze())
    #         loss += torch.mean(torch.square(out[:,i].squeeze()[~mask] - y[:,i].squeeze()[~mask]))
    else:
        raise('wrong estimator selected')
    
    return loss

class Dataset(torch.utils.data.Dataset):
  'Characterizes a dataset for PyTorch'
  def __init__(self, setting, data, labels, t):
        'Initialization'
        print(setting, data.shape, labels.shape)
        self.labels = labels
        self.data = data
        self.t = t
        
  def __len__(self):
        return self.labels.shape[0]

  def __getitem__(self, index):
        x = self.data[index,:]
        y = self.labels[index,:]
        t = self.t[index,:]

        return [x, y, t]


class Datasetw(torch.utils.data.Dataset):
  'Characterizes a dataset for PyTorch'
  def __init__(self, setting, data, labels, t, weights):
        'Initialization'
        print(setting, data.shape, labels.shape)
        self.labels = labels
        self.data = data
        self.t = t
        self.weights = weights
        
  def __len__(self):
        return self.labels.shape[0]

  def __getitem__(self, index):
        x = self.data[index,:]
        y = self.labels[index]
        t = self.t[index]
        w = self.weights[index]

        return [x, y, t, w]


def get_loader_from(X_train, X_val, y_train, y_val, t_train, t_val, device, batch, w_train=None, w_val=None, test_size=0.20):
    # if weight is None:
    #     X_train, X_val, y_train, y_val, = train_test_split(X, y, test_size=test_size, random_state=42, stratify=X[:,-1:].squeeze())
    # else:
    #     X_train, X_val, y_train, y_val, w_train, w_val = train_test_split(X, y, weight, test_size=test_size, random_state=42, stratify=X[:,-1:].squeeze())
    
    # convert to Tensor
    X_train = torch.Tensor(X_train).to(device)
    y_train = torch.Tensor(y_train).to(device)
    t_train = torch.Tensor(t_train).to(device)

    X_val = torch.Tensor(X_val).to(device)
    y_val = torch.Tensor(y_val).to(device)
    t_val = torch.Tensor(t_val).to(device)

    if w_train is not None:
        w_train = torch.Tensor(w_train).to(device)
        w_val = torch.Tensor(w_val).to(device)
        # weight = torch.Tensor(weight).to(device)

    # X = torch.Tensor(X).to(device)
    # y = torch.Tensor(y).to(device)

    if w_train is None:
        training_set = Dataset('Train', X_train, y_train, t_train)
        val_set = Dataset('Val', X_val, y_val, t_val)
        # full = Dataset('Full', X, y, t)
    else:
        training_set = Datasetw('Train', X_train, y_train, t_train, w_train)
        val_set = Datasetw('Val', X_val, y_val, t_val, w_val)
        # full = Datasetw('Full', X, y, t, weight)

    train_loader = torch.utils.data.DataLoader(training_set, batch_size=batch, shuffle=True) #, collate_fn=collate_batch
    val_loader = torch.utils.data.DataLoader(val_set, batch_size=batch, shuffle=False)
    # full_loader = torch.utils.data.DataLoader(full, batch_size=batch, shuffle=False)
    
    return train_loader, val_loader#, full_loader