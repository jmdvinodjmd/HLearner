'''
    Script to run meta-learner experiments
'''
# from cmath import exp
# from math import gamma
import wandb
import os
from tqdm import tqdm
import sys
# import matplotlib.pyplot as plt
# import time
# import datetime
# import argparse
import numpy as np
# import pandas as pd
import pickle
import random
# from random import SystemRandom
from sklearn.model_selection import train_test_split

import torch
import torch.nn as nn
import torch.nn.functional as F

import src.utils as utils
from src.data.load_dataset import get_data_loaders, load_data
from src.load_model_data import create_model, get_dataset

import warnings
# Suppress all warnings
warnings.filterwarnings("ignore")

############### COMMON SETUP ###############
args = utils.init_args()

# log_interval = 100
# start = time.time()

# selecting device
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print('\nDEVICE:', device)

# setting seed for reproducibility
utils.seed_torch(args.seed)

# load data
# load dataset details
file_name, input_size, num_cause, num_tasks, tasks = get_dataset(dataset=args.dataset)
args.num_cause = num_cause
args.name = args.model +'-'+args.name#args.dataset+'-'+str(args.seed)#+'-'+str(args.emb_dim_t)

utils.makedirs(args.save)
experimentID = args.load
if experimentID is None:
    experimentID = args.model + '-' + args.name  #int(SystemRandom().random()*100000)

utils.makedirs('./results/checkpoints/')
utils.makedirs('./results/figures/')
utils.makedirs('./results/raw/')
utils.makedirs('./results/csv/')
# set logger
log_path = os.path.join("./results/logs/" + "exp_" + str(experimentID) + ".log")
utils.makedirs("./results/logs/")
logger = utils.get_logger(logpath=log_path, filepath=os.path.abspath(__file__), displaying=True)
logger.info("Experiment " + str(experimentID))

for i in tqdm([0]): #range(repetitions)
    logger.info('rep:' + str(i))
    # initilising wandb
    # wandb.init(project=args.project, entity="jmdvinodjmd", reinit=True)
    wandb.init(mode="disabled")
    logger.info('\nRun..................>>>>>>>>' + args.model + '-' + args.dataset + str(args.data_size) + '-' + str(i) + '-'+ str(args.seed))
    logger.info('args:\n')
    logger.info(args)
    logger.info(sys.argv)
    wandb.run.name = args.name #+ str(args.data_size)+ str(args.emb_dim1)#+ str(args.hypernet1) + args.exp_name#+ '-'+ str(args.random_seed) #+ '-NV'+ str(clip_norm_tag)+str(clip_val_tag)+str(clip_value)
    wandb.config = vars(args)
 
    with open(file_name, 'rb') as f:
        X, T, Y, Y_cf, T_ind, tasks = pickle.load(f)

        # # we will remove values corresponding to missing outcomes
        # mask = ~np.isnan(Y[:,0])  # True for non-NaN, False for NaN
        # # Step 2: Filter X, T, Y, and Y_cf using the mask
        # X = X[mask]
        # T = T[mask]
        # Y = Y[mask]
        # Y_cf = Y_cf[mask]
        
        # X, T, Y, Y_cf = X[:200+700,:], T[:200+700,:], Y[:200+700,:], Y_cf[:200+700,:,:]

        # Y = np.expand_dims(Y, axis=1)
        # Y_cf = np.expand_dims(Y_cf, axis=1)

    print('Loaded:', X.shape, T.shape, Y.shape, Y_cf.shape)#, T_ind)
    # input_size = X.shape[1]

    # set dataset characteristics
    args.N = X.shape[0]
    args.num_task = Y.shape[1]
    args.emb_dim1 = args.num_cause = T.shape[1]
    args.tasks=tasks

    # X_train, X_test, y_train, y_test, y_cftrain, y_cftest, t_train, t_test = train_test_split(X, Y, Y_cf, T, test_size=int(args.N*0.30), 
    #                                                                      random_state=42, stratify=Y[:,0] + Y[:,0]*T[:,0])
    X_train, X_test, y_train, y_test, y_cftrain, y_cftest, t_train, t_test = train_test_split(X, Y, Y_cf, T, test_size=args.test_size, 
                                                                         random_state=42) # , stratify=T[:,-1]
    print('\n\nDataset:\nX_train:', X_train.shape, ' X_test:', X_test.shape)
    # # scale experiments
    # if args.data_size:
    #     logger.info('\nRunning scale experiments : ' + str(args.data_size))
    #     X_full, y_full, t_full, mu0_full, mu1_full = X_full[:args.data_size,:], y_full[:args.data_size], t_full[:args.data_size], mu0_full[:args.data_size], mu1_full[:args.data_size]
    if args.tuning==1 or utils.get_best_param(args.model + '-' + args.dataset, 'best_params.json') is None:
        param_list = utils.get_scp_config(args)
        print('\n------Hyperparameter tuning---------')
        print(param_list)
        tuning_results = {}
        for i, params in enumerate(param_list):
            print(i, ' params', params)
            args.lr1 = params.learning_rate
            args.batch_size = params.batch_size
            # HLearner has same structure as SLearner - so need to have hidden
            if args.model == 'HLearner':
                args.emb_dim_t = params.emb_dim_t
                args.emb_dim_y = params.emb_dim_y
                if utils.get_best_param('SLearner' + '-' + args.dataset, 'best_params.json') is not None:
                    args.hidden_size = utils.get_best_param('SLearner' + '-' + args.dataset, 'best_params.json')[1]
                elif utils.get_best_param('xSLearner' + '-' + args.dataset, 'best_params.json') is not None:
                    args.hidden_size = utils.get_best_param('xSLearner' + '-' + args.dataset, 'best_params.json')[1]
                else:
                    assert False, 'HLearner cant find hidden_size param.'
            else:
                args.hidden_size = params.hidden_size
            
            ########## temporary value based on SCP
            args.hidden_size = X_train.shape[1] + t_train.shape[1] + 1

            model = create_model(args, input_size, y_train.shape[1], device)
            logger.info(model)
            results_dict = model.fit(X_train, y_train, t_train, logger, wandb, device)
            tuning_results[(int(args.batch_size),int(args.hidden_size),float(args.lr1),int(args.emb_dim_y),int(args.emb_dim_t))] = results_dict['Best Val Loss']

        print(tuning_results)
        utils.save_best_params(tuning_results, args.model + '-' + args.dataset, 'best_params.json')          

    elif args.tuning==0:
        print('\n Loading best hyperparameters---------')
        params = utils.get_best_param(args.model + '-' + args.dataset, 'best_params.json')   
        args.batch_size = params[0]
        args.hidden_size = params[1]
        args.lr1 = params[2]
        args.emb_dim_y = params[3]
        args.emb_dim_t = params[4]
    else:
        print('----------using default parameters------')
    
    ########## temporary value based on SCP ##################
    args.hidden_size = X_train.shape[1] + t_train.shape[1] + 1

    model = create_model(args, input_size, y_train.shape[1], device)
    logger.info(model)
    print('\n------Training model---------')
    results_dict = model.fit(X_train, y_train, t_train, logger, wandb, device)

    print('\n------Evaluating the factual model---------')
    # Train Evaluation
    results_dict = utils.evaluate_factual_model(model, X_train, t_train, y_train, device, args, wandb, logger, results_dict, dataset_type='Train')
    # Test Evaluation
    results_dict = utils.evaluate_factual_model(model, X_test, t_test, y_test, device, args, wandb, logger, results_dict, dataset_type='Test')
    
    print('\n------Evaluating the counterfactual model-------')
    results_dict = utils.evaluate_counterfactual_model(model, X, X_train, X_test, Y_cf, y_cftrain, y_cftest, T, t_train, t_test, T_ind, 
                                  results_dict, args, logger, wandb, device)

utils.save_dict_to_excel('./results/csv/'+args.name +'.xlsx', results_dict)
logger.info('\nparameters:')
logger.info(args)
logger.info('...........Experiment ended.............')
#########################################################