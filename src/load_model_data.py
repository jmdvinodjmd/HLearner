import argparse, logging
import os, pickle
# from types import _StaticFunctionType
import numpy as np
import torch
import matplotlib.pyplot as plt
from torch import nn
import pandas as pd

from src.models.hypernet_estimators import *
from src.models.baseline_estimators import *
# from src.models.hypernetworks import HyperNLearner


def create_model(args, input_size, num_task, device):
    if args.model =='SLearner' or args.model =='xSLearner':
        target_layers=[[args.hidden_size]]
        for _ in range(args.num_task):
            target_layers.append([1])
        # print(target_layers, args.tasks, len(target_layers)-1, len(args.tasks))
        # assert False
        activations, hn_target_activations = utils.get_activations_config(target_layers, args.tasks)
        assert len(args.tasks) == num_task, 'number of tasks must match'
        model = SLearner(target_layers, activations, args, input_size, args.num_cause, device)
    elif args.model == "HLearner":
        target_layers=[[args.hidden_size], [1]]
        tasks=['binary'] # this does not matter
        activations, hn_target_activations = utils.get_activations_config(target_layers, tasks)
        model = HLearner(target_layers, hn_target_activations, args, input_size, device)
    else:
        raise Exception('incorrect method selected...', args.model)    

    return model


def get_dataset(dataset, p=0.1, tr_size=100):
    binary = 0
    if dataset =="MCMO":
        file_name = './data/MCMO.pkl'
        with open(file_name, 'rb') as f:
            X, T, Y, Y_cf, T_ind, tasks = pickle.load(f)
            input_size = X.shape[1]
            num_causes = T.shape[1]
            if Y.ndim>1:
                num_tasks = Y.shape[1]
            else:
                num_tasks = 1
                
    elif dataset =="MCMO-NB1200":
        file_name = './data/NB/'+dataset+'.pkl'
    elif dataset =="MCMO-NB1700":
        file_name = './data/NB/'+dataset+'.pkl'
    elif dataset =="MCMO-NB1000":
        file_name = './data/NB/'+dataset+'.pkl'
    elif dataset =="MCMO-NB2700":
        file_name = './data/NB/'+dataset+'.pkl'

    elif dataset =="MCMO-TB2":
        file_name = './data/TB/'+dataset+'.pkl'
    elif dataset =="MCMO-TB4":
        file_name = './data/TB/'+dataset+'.pkl'
    elif dataset =="MCMO-TB8":
        file_name = './data/TB/'+dataset+'.pkl'
    elif dataset =="MCMO-TB10":
        file_name = './data/TB/'+dataset+'.pkl'  
    elif dataset =="MCMO-TB12":
        file_name = './data/TB/'+dataset+'.pkl'  
    
    elif dataset =="MCMO-YB1":
        file_name = './data/YB/'+dataset+'.pkl'
    elif dataset =="MCMO-YB2":
        file_name = './data/YB/'+dataset+'.pkl'
    elif dataset =="MCMO-YB3":
        file_name = './data/YB/'+dataset+'.pkl'
    elif dataset =="MCMO-YB4":
        file_name = './data/YB/'+dataset+'.pkl' 
    elif dataset =="MCMO-YB10":
        file_name = './data/YB/'+dataset+'.pkl' 

    elif dataset =="MCMO-DB10":
        file_name = './data/DB/'+dataset+'.pkl'
    elif dataset =="MCMO-DB20":
        file_name = './data/DB/'+dataset+'.pkl'
    elif dataset =="MCMO-DB30":
        file_name = './data/DB/'+dataset+'.pkl'
    elif dataset =="MCMO-DB40":
        file_name = './data/DB/'+dataset+'.pkl'
    
    elif dataset =="MCMO-CB0.5":
        file_name = './data/CB/'+dataset+'.pkl'
    elif dataset =="MCMO-CB1.0":
        file_name = './data/CB/'+dataset+'.pkl'
    elif dataset =="MCMO-CB2":
        file_name = './data/CB/'+dataset+'.pkl'
    elif dataset =="MCMO-CB4":
        file_name = './data/CB/'+dataset+'.pkl'

    # continuous treatments
    elif dataset =="MCMO-NC1200":
        file_name = './data/NC/'+dataset+'.pkl'
    elif dataset =="MCMO-NC1700":
        file_name = './data/NC/'+dataset+'.pkl'
    elif dataset =="MCMO-NC1000":
        file_name = './data/NC/'+dataset+'.pkl'
    elif dataset =="MCMO-NC2700":
        file_name = './data/NC/'+dataset+'.pkl'   

    elif dataset =="MCMO-TC2":
        file_name = './data/TC/'+dataset+'.pkl'
    elif dataset =="MCMO-TC4":
        file_name = './data/TC/'+dataset+'.pkl'
    elif dataset =="MCMO-TC8":
        file_name = './data/TC/'+dataset+'.pkl'
    elif dataset =="MCMO-TC10":
        file_name = './data/TC/'+dataset+'.pkl'   

    elif dataset =="MCMO-YC1":
        file_name = './data/YC/'+dataset+'.pkl'
    elif dataset =="MCMO-YC2":
        file_name = './data/YC/'+dataset+'.pkl'
    elif dataset =="MCMO-YC3":
        file_name = './data/YC/'+dataset+'.pkl'
    elif dataset =="MCMO-YC4":
        file_name = './data/YC/'+dataset+'.pkl' 
    elif dataset =="MCMO-YC10":
        file_name = './data/YC/'+dataset+'.pkl'

    elif dataset =="MCMO-DC10":
        file_name = './data/DC/'+dataset+'.pkl'
    elif dataset =="MCMO-DC20":
        file_name = './data/DC/'+dataset+'.pkl'
    elif dataset =="MCMO-DC30":
        file_name = './data/DC/'+dataset+'.pkl'
    elif dataset =="MCMO-DC40":
        file_name = './data/DC/'+dataset+'.pkl'

    elif dataset =="MCMO-CC0.5":
        file_name = './data/CC/'+dataset+'.pkl'
    elif dataset =="MCMO-CC1.0":
        file_name = './data/CC/'+dataset+'.pkl'
    elif dataset =="MCMO-CC2":
        file_name = './data/CC/'+dataset+'.pkl'
    elif dataset =="MCMO-CC4":
        file_name = './data/CC/'+dataset+'.pkl'

    elif dataset =="IHDP":
        input_size = 25+1
        file_train = '../SNet+/data/ihdp_npci_1-100.train.npz'
        file_test =  '../SNet+/data/ihdp_npci_1-100.test.npz'
        repetitions = 100
    elif dataset =="acic":
        input_size = 55+1
        file_train = '../SNet+/data/acic2016-train.npz'
        file_test = '../SNet+/data/acic2016-test.npz'
        repetitions = 1
    elif dataset =="jobs":
        input_size = 17+1
        file_train = '../SNet+/data/jobs_DW_bin.new.10.train'
        file_test = '../SNet+/data/jobs_DW_bin.new.10.test.npz'
        repetitions = 10
    elif dataset =="twins":
        input_size = 39+1
        file_train = '../SNet+/data/twins/twins-t-'+str(p)+'-tr-'+str(tr_size)+'-train.npz'
        file_test = '../SNet+/data/twins/twins-t-'+str(p)+'-tr-'+str(tr_size)+'-test.npz'
        repetitions = 1
        binary = 1
    else:
        raise Exception('please select correct dataset - ' + dataset)
    
    with open(file_name, 'rb') as f:
        X, T, Y, Y_cf, T_ind, tasks = pickle.load(f)
        input_size = X.shape[1]
        num_causes = T.shape[1]
        if Y.ndim>1:
            num_tasks = Y.shape[1]
        else:
            num_tasks = 1
    
    return file_name, input_size, num_causes, num_tasks, tasks