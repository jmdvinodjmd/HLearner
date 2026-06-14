import argparse, logging
from typing import NamedTuple
import os, pickle
# from types import _StaticFunctionType
import numpy as np
import torch
import torch.nn.functional as F
import matplotlib.pyplot as plt
from torch import nn
import pandas as pd
from sklearn.metrics import roc_auc_score
from sklearn import metrics
from sklearn.model_selection import train_test_split


def init_args():
    parser = argparse.ArgumentParser('ITE')
    parser.add_argument('--niters', type=int, default=1000)
    parser.add_argument('--print', type=int, default=1)
    parser.add_argument('--print-every', type=int, default=50)
    parser.add_argument('--nfold', type=int, default=1) 
    parser.add_argument('--clip-value', type=float, default=1)
    parser.add_argument('--clip-val-tag', type=int, default=0)
    parser.add_argument('--clip-norm-tag', type=int, default=0) 
    
    # parser.add_argument('--ortho-reg',  type=float, default=0, help="ortho_regularisation factor.")
    parser.add_argument('--lr1',  type=float, default=1e-04, help="learning rate for plug-in approach or for the first stage")
    parser.add_argument('--lr2',  type=float, default=1e-04, help="learning rate.")
    parser.add_argument('--weight-decay',  type=float, default=1e-04, help="weight_decay")
    parser.add_argument('-b', '--batch-size', type=int, default=100)
    parser.add_argument('--patience', type=int, default=10)
    parser.add_argument('--model', type=str, default='HLearner', help="SLearner, TLearner, ")
    parser.add_argument('--name', type=str, default='', help="SLearner, TLearner, ")
    parser.add_argument('--project', type=str, default='ITE-test', help="SLearner, TLearner, ")
    parser.add_argument('--dataset', type=str, default='MCMO', help="IHDP, syn3000, syn10000")
    parser.add_argument('--save', type=str, default='results/', help="Path for save checkpoints")
    parser.add_argument('--load', type=str, default=None, help="ID of the experiment to load for evaluation. If None, run a new experiment.")
    parser.add_argument('-r', '--seed', type=int, default=0, help="Random_seed")

    parser.add_argument('--spect-norm1', type=int, default=1)
    parser.add_argument('--spect-norm2', type=int, default=1)
    parser.add_argument('--num-chunks', type=int, default=10)
    parser.add_argument('--drop-rate',  type=float, default=0.0, help="dropout rate")
    parser.add_argument('--hn-drop-rate1',  type=float, default=0.0, help="dropout rate in HN1.")
    parser.add_argument('--hn-drop-rate2',  type=float, default=0.0, help="dropout rate in HN2.")
    parser.add_argument('--hypernet1', type=str, default='all', help="layerwise, chunking, all")
    parser.add_argument('--hypernet2', type=str, default='all', help="layerwise, chunking, all")

    parser.add_argument('--val-size',  type=int, default=200, help="")
    parser.add_argument('--test-size',  type=int, default=500, help="")
    parser.add_argument('--N', type=int, default=100)
    parser.add_argument('--num-cause', type=int, default=2, help="number of causes")
    parser.add_argument('--num-task', type=str, default='HLearner', help="specify")
    parser.add_argument('--tasks', type=str, default="['binary', 'cont']", help="specify tasks in order of index in Y")
    parser.add_argument('--emb-dim-t', type=int, default=16, help="embedding dimension for treatments")
    parser.add_argument('--emb-dim-y', type=int, default=8, help="embedding dimension for tasks")
    parser.add_argument('--data-size',  type=int, default=0, help="for scale experiments.")
    parser.add_argument("--tuning", type=int, default=0)
    parser.add_argument('--tuning-iters', type=int, default=5)
    parser.add_argument('--hidden-size', type=int, default=32)

    # for twins dataset
    parser.add_argument('--p',  type=float, default=0.1, help="probability of selecting twin")
    parser.add_argument('--n',  type=str, default=500, help="for scale experiments.")

    args = parser.parse_args()

    return args

def makedirs(dirname):
    if not os.path.exists(dirname):
        os.makedirs(dirname)

def save_checkpoint(state, save, epoch):
    if not os.path.exists(save):
        os.makedirs(save)
    filename = os.path.join(save, 'checkpt-%04d.pth' % epoch)
    torch.save(state, filename)


def get_logger(logpath, filepath, package_files=[],
               displaying=True, saving=True, debug=False):
    logger = logging.getLogger()
    if debug:
        level = logging.DEBUG
    else:
        level = logging.INFO
    logger.setLevel(level)
    if (logger.hasHandlers()):
        logger.handlers.clear()
    if saving:
        info_file_handler = logging.FileHandler(logpath, mode='w')
        info_file_handler.setLevel(level)
        logger.addHandler(info_file_handler)
    if displaying:
        console_handler = logging.StreamHandler()
        console_handler.setLevel(level)
        logger.addHandler(console_handler)
    logger.info(filepath)

    for f in package_files:
        logger.info(f)
        with open(f, 'r') as package_f:
            logger.info(package_f.read())

    return logger

def calc_auroc(y_true, y_prob):
    import sklearn
    return sklearn.metrics.roc_auc_score(y_true, y_prob)

def calc_metrics(args, out, Y, mode, wandb, normalization_factor, logger, task='cont'):
    out = out.detach().cpu().numpy().reshape(-1)
    Y = Y.reshape(-1)
    mask = np.isnan(Y)
    mse = np.mean(np.square(out[~mask] - Y[~mask]))
    if task != 'cont':
        auc = roc_auc_score(Y[~mask], out[~mask])
    else:
        auc = 0
    # ate_pred = ate_pred.detach().cpu().numpy()
    # te_pred = te_pred.detach().cpu().numpy()

    # pehe = np.mean(np.square((te_true - te_pred)))
    # sqrt_pehe = np.sqrt(pehe)

    # logger.info("PEHE:"+'-'+mode +': '+ str(sqrt_pehe) +" ATE:"+'-'+mode +': '+ str(ate_pred))
    # wandb.log({"PEHE"+'_'+mode: sqrt_pehe})
    # wandb.log({"ATE"+'_'+mode: ate_pred})
    
    return [mse, auc]

def plot_cates(args, y_true, y0_pred, y1_pred, mu0, mu1, props_true, model, mode, wandb, normalization_factor, logger):
    cate_pred=(y1_pred-y0_pred).squeeze()
    cate_true=(mu1-mu0).squeeze()
    ate_pred=torch.mean(cate_pred) # taking mean of absolute for ATE
    ate_true=torch.mean(cate_true)
    ate = ate_true-ate_pred

    cate_pred = cate_pred.detach().cpu().numpy()
    ate = ate.detach().cpu().numpy()
    cate_true = cate_true.detach().cpu().numpy()
    
    if args.binary:
        y1_pred = y1_pred.detach().cpu().numpy().reshape(-1)
        y0_pred = y0_pred.detach().cpu().numpy().reshape(-1)
        mu0 = mu0.detach().cpu().numpy().reshape(-1)
        mu1 = mu1.detach().cpu().numpy().reshape(-1)
        
        auc_ite, auc_mu0, auc_mu1, ap_mu0, ap_mu1 = evaluate_ITE_binary(mu0, mu1, y0_pred, y1_pred)

        logger.info("\nAUROC-ITE:"+'_'+mode +': '+ str(auc_ite))
        logger.info("\nAUROC-0:"+'_'+mode +': '+ str(auc_mu0))
        logger.info("\nAUROC-1:"+'_'+mode +': '+ str(auc_mu1))
        logger.info("\nAUPRC-0:"+'_'+mode +': '+ str(ap_mu0))
        logger.info("\nAUPRC-1:"+'_'+mode +': '+ str(ap_mu1))
        wandb.log({"AUCROC-ITE"+'_'+mode: auc_ite})
        wandb.log({"AUROC-0"+'_'+mode: auc_mu0})
        wandb.log({"AUROC-1"+'_'+mode: auc_mu1})
        wandb.log({"AUPRC-0"+'_'+mode: ap_mu0})
        wandb.log({"AUPRC-1"+'_'+mode: ap_mu1})

    # fig = plt.figure()
    # pd.Series(cate_pred).plot.kde(color='blue')
    # pd.Series(cate_true).plot.kde(color='green')
    # ax = pd.Series(cate_true-cate_pred).plot.kde(color='red')
    # ax.legend(['CATE-Pred','CATE-True','Error'])
    # ax.set_title(model)
    # fig.savefig('./results/figures/' + model+'_'+mode + '.png')

    pehe = np.mean( np.square( ( cate_true - cate_pred) ) )
    sqrt_pehe = np.sqrt(pehe)
    # we report RMSE normalized by standard deviation of the observed factual training data.....
    # sqrt_pehe = np.sqrt(pehe) #/ normalization_factor.detach().cpu().numpy()

    logger.info("PEHE:"+'-'+mode +': '+ str(sqrt_pehe) +" ATE:"+'-'+mode +': '+ str(ate))
    wandb.log({"PEHE"+'-'+mode: sqrt_pehe})
    # wandb.log({"ATE"+'-'+mode: ate})
    
    return sqrt_pehe, ate

def evaluate_ITE_binary(y0_out, y1_out, mu0_pred, mu1_pred):
    from sklearn.preprocessing import label_binarize
    from sklearn.metrics import roc_auc_score, average_precision_score

    ite_out = y1_out - y0_out
    ite_out_encoded = label_binarize(ite_out, classes=[-1, 0, 1])

    # create probabilities for each possible level of ITE
    probs = np.zeros((len(y1_out), 3))
    probs[:, 0] = (mu0_pred * (1 - mu1_pred)).reshape((-1,))  # P(Y1-Y0=-1)
    probs[:, 1] = ((mu0_pred * mu1_pred) + ((1 - mu0_pred) * (1 - mu1_pred))).reshape((-1,))  # P(Y1-Y0=0)
    probs[:, 2] = (mu1_pred * (1 - mu0_pred)).reshape((-1,))  # P(Y1-Y0=1)
    auc_ite = roc_auc_score(ite_out_encoded, probs)

    # evaluate performance on potential outcomes
    auc_mu0 = roc_auc_score(y0_out, mu0_pred)
    auc_mu1 = roc_auc_score(y1_out, mu1_pred)
    ap_mu0 = average_precision_score(y0_out, mu0_pred)
    ap_mu1 = average_precision_score(y1_out, mu1_pred)

    return auc_ite, auc_mu0, auc_mu1, ap_mu0, ap_mu1


# https://github.com/Bjarten/early-stopping-pytorch/blob/master/MNIST_Early_Stopping_example.ipynb
import numpy as np
import torch

class EarlyStopping:
    """Early stops the training if validation loss doesn't improve after a given patience."""
    def __init__(self, patience=7, verbose=False, delta=0, path='checkpoint.pt', trace_func=print, logger=None):
        """
        Args:
            patience (int): How long to wait after last time validation loss improved.
                            Default: 7
            verbose (bool): If True, prints a message for each validation loss improvement.
                            Default: False
            delta (float): Minimum change in the monitored quantity to qualify as an improvement.
                            Default: 0
            path (str): Path for the checkpoint to be saved to.
                            Default: 'checkpoint.pt'
            trace_func (function): trace print function.
                            Default: print
        """
        self.patience = patience
        self.verbose = verbose
        self.counter = 0
        self.best_score = None
        self.early_stop = False
        self.val_auc_max = np.Inf
        self.delta = delta
        self.path = path
        # self.trace_func = trace_func
        self.logger = logger

    def __call__(self, val_auc, model):

        score = - val_auc

        if self.best_score is None:
            self.best_score = score
            self.save_checkpoint(val_auc, model)
        elif score < self.best_score + self.delta:
            self.counter += 1
            self.logger.info(f'EarlyStopping counter: {self.counter} out of {self.patience}')
            if self.counter >= self.patience:
                self.early_stop = True
        else:
            self.best_score = score
            self.save_checkpoint(val_auc, model)
            self.counter = 0

    def save_checkpoint(self, val_auc, model):
        '''Saves model when validation acu increase.'''
        if self.verbose:
            self.logger.info(f'Validation auc increased ({self.val_auc_max:.6f} --> {val_auc:.6f}).  Saving model ...')
        torch.save(model.state_dict(), self.path)
        self.val_auc_max = val_auc

###############################################################################
###### Hypernet utilities ##############
###############################################################################
def weighted_mse_loss(input, target, weight=None):
    if weight is None:
        return torch.mean((input - target) ** 2)
    else:
        return torch.mean(weight * (input - target) ** 2)

def weighted_binary_cross_entropy(pred, y, weight=None):
    loss = torch.nn.BCELoss(reduction='none')
    if weight is None:
        return torch.mean(loss(pred, y))
        # return -torch.mean(pred.log()*y + (1-y)*(1-pred).log())
    else:
        return torch.mean(weight * loss(pred, y))
        # return -torch.mean(weight * (pred.log()*y + (1-y)*(1-pred).log()))

###############################################################################
# def init_weights(m):
#     if isinstance(m, nn.Linear):
#         # torch.nn.init.xavier_normal_(m.weight)
#         # torch.nn.init.xavier_uniform_(m.weight)
#         # torch.nn.init.kaiming_normal_(m.weight)
#         # torch.nn.init.kaiming_uniform_(m.weight)
#         # m.bias.data.fill_(0.01)
#         # nn.init.normal_(m.weight, 0, 0.01)
#         # nn.init.normal_(m.bias, 0, 0.01)
#         nn.init.zeros_(m.bias)

#     # if isinstance(m, nn.Embedding):
#     #     # nn.init.normal_(m.weight.data, mean=0, std=0.2)
#     #     # torch.nn.init.xavier_normal_(m.weight)
#     #     torch.nn.init.xavier_uniform_(m.weight)
#     #     # torch.nn.init.kaiming_normal_(m.weight)

############################################################
class MLP_Model(nn.Module):
    def __init__(self, net_layers, activations, input_size, t_dim, dropout_rate=0.0):
        super(MLP_Model, self).__init__()
        # create hypernetwork layers
        self.phi = nn.ModuleList()
        # adding feature extracter layers
        for i in range(len(net_layers[0])):
            if i == 0:
                self.phi.append(nn.Linear(input_size+t_dim, net_layers[0][i]))
                self.phi.append(activations[0][i]())
                self.phi.append(nn.Dropout(dropout_rate))
            else:
                self.phi.append(nn.Linear(net_layers[0][i-1], net_layers[0][i]))
                self.phi.append(activations[0][i]())
                self.phi.append(nn.Dropout(dropout_rate))
        
        # adding the task specific layers
        self.tasks = nn.ModuleList()
        for i in range(len(net_layers[1:])):
            task = nn.ModuleList()
            for j in range(len(net_layers[1+i])):
                if j==0:
                    task.append(nn.Linear(net_layers[0][-1], net_layers[i+1][j]))
                    if activations[i+1][j] is not None:
                        task.append(activations[i+1][j]())
                else:
                    task.append(nn.Linear(net_layers[i+1][j-1], net_layers[i+1][j]))
                    if activations[i+1][j] is not None:
                        task.append(activations[i+1][j]())
            
            self.tasks.append(task)

        # # initialise weights
        # self.apply(init_weights)
    
    def forward(self, X, t):
        # combine X and t
        out = torch.cat((X, t), dim=1)

        # extract features
        for layer in self.phi:
            out = layer(out)

        # task specific predictions
        outputs = []
        # print('size------------------', len(self.tasks))
        for i in range(len(self.tasks)):
            temp = out
            for layer in self.tasks[i]:
                temp = layer(temp)
            
            outputs.append(temp.squeeze())

        return torch.stack(outputs, dim=1)
    

         
def seed_torch(seed=0):
    import random

    random.seed(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed) # if you are using multi-GPU.
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True

################################

def copy_activations(activations):
    acts = []
    for i in range(2):
        a = []
        for j in range(2):
            a.append(activations[i][j].copy())
        acts.append(a)
    
    return acts


############
def get_activations_config(target_layers=[[32, 16], [8, 1], [8, 1]], 
                           tasks=['cont', 'cont']):
    print(target_layers, tasks)
    assert len(target_layers)-1 == len(tasks), 'number of tasks not matching'

    mapping1 = {'cont':None, 'binary':nn.Sigmoid}
    mapping2 = {'cont':None, 'binary':F.sigmoid}
    activations = []
    hn_target_activations = []

    for i in range(len(target_layers)):
        a1 = []
        a2 = []
        for j in range(len(target_layers[i])):
            if (i>0) and (j == len(target_layers[i])-1):
                a1.append(mapping1[tasks[i-1]])
                a2.append(mapping2[tasks[i-1]])   
            else:
                a1.append(nn.ReLU)
                a2.append(F.relu)
        
        activations.append(a1)
        hn_target_activations.append(a2)

    return activations, hn_target_activations

# def get_activations_config_hn(target_layers=[[32, 16], [8, 1]], 
#                            tasks=['cont', 'cont']):
#     assert len(target_layers)-1 == len(tasks), 'number of tasks not matching'

#     mapping2 = {'cont':None, 'binary':F.sigmoid}
#     activations = []
#     hn_target_activations = []

#     for i in range(len(target_layers)):
#         a1 = []
#         a2 = []
#         for j in range(len(target_layers[i])):
#                 a2.append(F.relu)
        
#         activations.append(a1)
#         hn_target_activations.append(a2)
#     activations[-1][-1:] = nn.Sigmoid
#     hn_target_activations[-1][-1:] = F.sigmoid

#     return activations, layers

##################################
def save_results_dict(file_name, results_dict):
    with open(file_name, 'wb') as f:
        pickle.dump(results_dict, f)

def load_results_dict(file_name):
    with open(file_name, 'rb') as f:
        results_dict = pickle.load(f)
    
    return results_dict
def dict_to_file(name, results_sizes):
    '''
    Convert the nested dictionary to a dataframe and save
    '''
    flat_data = []
    for size, size_dict in results_sizes.items():
        for repeat, repeat_dict in size_dict.items():
            repeat_dict['Size'] = size
            repeat_dict['Repeat'] = repeat
            flat_data.append(repeat_dict)

    # Convert Results into Dataframe and save
    df = pd.DataFrame(flat_data)
    # Save DataFrame to Excel
    df.to_excel(name, index=False)

def combine_dicts(dict_list):
    combined_dict = {}

    for d in dict_list:
        for key, value in d.items():
            if key not in combined_dict:
                # If the key is not in the combined dictionary, initialize it
                combined_dict[key] = value if isinstance(value, list) else [value]
            else:
                # If the key exists, concatenate lists or append values
                if isinstance(value, list):
                    combined_dict[key].extend(value)
                else:
                    combined_dict[key].append(value)
    # rename the result fields and create new one with sum of those columns
    new_dict = {}
    for key, value in combined_dict.items():
        if key == 'CounterFactual Train MSE':
            new_dict['CounterFactual Train MSE-all tasks'] = value
            new_dict['CounterFactual Train MSE'] = sum(value)
        elif key == 'CounterFactual Test MSE':
            new_dict['CounterFactual Test MSE-all tasks'] = value
            new_dict['CounterFactual Test MSE'] = sum(value)
        elif key == 'CounterFactual All MSE':
            new_dict['CounterFactual All MSE-all tasks'] = value
            new_dict['CounterFactual All MSE'] = sum(value)
        else:
            new_dict[key] = value

    return new_dict

def save_dict_to_excel(file_name, data_dict):
    # Ensure all values in the dictionary are lists
    for key in data_dict:
        if not isinstance(data_dict[key], list):
            data_dict[key] = [data_dict[key]]
    
    # Find the maximum length of lists in the dictionary
    max_len = max(len(v) for v in data_dict.values())
    
    # Pad shorter lists with None (or NaN)
    padded_dict = {k: v + [None]*(max_len - len(v)) for k, v in data_dict.items()}
    
    # Convert dictionary to pandas DataFrame
    df = pd.DataFrame(padded_dict)
    
    # Save DataFrame to Excel file
    df.to_excel(file_name, index=False)

def save_mse_to_excel(dict_list, file_name):
    """
    Converts a list of dictionaries into a DataFrame and saves it to an Excel file.
    
    Parameters:
    - dict_list (list of dict): List of dictionaries with MSE values.
    - file_name (str): Path to save the Excel file.
    
    The dictionaries should have the following keys:
    - 'CF Train MSE': Counterfactual Train MSE value.
    - 'CF Test MSE': Counterfactual Test MSE value.
    """
    # Convert list of dictionaries into a DataFrame
    df = pd.DataFrame(dict_list)
    
    # Calculate the sum of the "CF Train MSE" and "CF Test MSE" columns
    train_mse_sum = df['CF Train MSE'].sum()
    test_mse_sum = df['CF Test MSE'].sum()
    print(float(train_mse_sum))
    # Add new columns for the summed values
    df['CounterFactual Train MSE'] = [float(train_mse_sum)] + [np.nan] * (len(df) - 1)
    df['CounterFactual Test MSE'] = [float(test_mse_sum)] + [np.nan] * (len(df) - 1)
    
    # Save the modified DataFrame to an Excel file
    df.to_excel(file_name, sheet_name='MSE Data', index=False)
    
    print(f"Data saved to {file_name} successfully.")


class HyperParameterConfig1(NamedTuple):
    itr: int
    learning_rate: float
    batch_size: int
    hidden_size: int

class HyperParameterConfig2(NamedTuple):
    itr: int
    learning_rate: float
    batch_size: int
    emb_dim_t: int
    emb_dim_y: int

def get_scp_config(args):
    param_list = list()

    if args.model == 'SLearner' or args.model == 'xSLearner':
        for i in range(args.tuning_iters):
            config = HyperParameterConfig1(
                itr=i,
                # hidden_size=np.random.randint(10, 40),
                hidden_size=26, #np.random.choice([16, 24, 32], size=1)[0],
                learning_rate=0.005, #np.random.choice([0.00002, 0.0001,  0.0005], size=1)[0],
                # batch_size=int(np.random.choice([50, 100, 200], size=1)[0]),
                # learning_rate=0.0001,
                batch_size=100,
                )
            param_list.append(config)
    elif args.model == 'HLearner' and args.dataset[6]=='B':
        # for i in range(args.tuning_iters):
        #     config = HyperParameterConfig2(
        #     itr=i,
        #     # hidden_size=np.random.randint(10, 40),
        #     learning_rate=0.005, #np.random.choice([0.00002, 0.0001,  0.0005], size=1)[0],
        #     # batch_size=int(np.random.choice([50, 100, 200], size=1)[0]),
        #     # learning_rate=0.0001,
        #     batch_size=100,
        #     emb_dim_t=np.random.choice([8, 16, 32], size=1)[0],
        #     emb_dim_y=4 #np.random.choice([4, 8, 16], size=1)[0],
        #     )
        #     param_list.append(config)
        param_list = [HyperParameterConfig2(itr=0, learning_rate=0.005, batch_size=100, emb_dim_t=8,emb_dim_y=4),
                           HyperParameterConfig2(itr=1, learning_rate=0.005, batch_size=100, emb_dim_t=16,emb_dim_y=4),
                           HyperParameterConfig2(itr=2, learning_rate=0.005, batch_size=100, emb_dim_t=32,emb_dim_y=4)]
    elif args.model == 'HLearner' and args.dataset[6]=='C':
        param_list = [HyperParameterConfig2(itr=0, learning_rate=0.005, batch_size=100, emb_dim_t=16,emb_dim_y=4),
                           HyperParameterConfig2(itr=1, learning_rate=0.005, batch_size=100, emb_dim_t=16,emb_dim_y=8),
                           HyperParameterConfig2(itr=2, learning_rate=0.005, batch_size=100, emb_dim_t=16,emb_dim_y=16),
                           HyperParameterConfig2(itr=3, learning_rate=0.005, batch_size=100, emb_dim_t=32,emb_dim_y=4),
                           HyperParameterConfig2(itr=4, learning_rate=0.005, batch_size=100, emb_dim_t=32,emb_dim_y=8),
                           HyperParameterConfig2(itr=5, learning_rate=0.005, batch_size=100, emb_dim_t=32,emb_dim_y=16)]
    return param_list

import json

def save_best_params(data_dict, json_key, file_name):

    # Find the key with the min value
    best_key = min(data_dict, key=data_dict.get)

    # Check if the JSON file exists
    if os.path.exists(file_name):
        # Load existing data from the JSON file
        with open(file_name, 'r') as json_file:
            json_data = json.load(json_file)
    else:
        # If the file doesn't exist, initialize an empty dictionary
        json_data = {}

    # Add/update the new key-value pair
    json_data[json_key] = list(best_key)

    # Save the updated data back to the JSON file
    with open(file_name, 'w') as json_file:
        json.dump(json_data, json_file, indent=4)
    
    print(f"Best key '{best_key}' added to JSON file '{file_name}' with key '{json_key}'.")

def get_best_param(json_key, file_name, default_value=None):
    # Check if the JSON file exists
    if os.path.exists(file_name):
        # Load the data from the JSON file
        with open(file_name, 'r') as json_file:
            json_data = json.load(json_file)
        
        # Check if the key exists in the JSON data
        if json_key in json_data:
            return json_data[json_key]
        else:
            print(f"Key '{json_key}' not found in '{file_name}'.")
            return default_value
    else:
        print(f"File '{file_name}' does not exist.")
        return default_value

def evaluate_factual_model(model, X_data, t_data, y_data, device, args, wandb, logger, results_dict, dataset_type='Train'):
    out = model.predict(X_data, t_data, device)
    mse = 0
    auc = 0
    results_dict['Factual '+dataset_type+' MSE'] = []
    results_dict['Factual '+dataset_type+' AUC'] = []
    for i in range(args.num_task):
        temp = calc_metrics(args, out[:,i], y_data[:,i], 'out', wandb, 1, logger, args.tasks[i])
        print(dataset_type+' MSE: {:.4f}, AUC: {:.4f}'.format(temp[0], temp[1]))
        mse += temp[0]
        auc += temp[1]
        # save results
        results_dict['Factual '+dataset_type+' MSE'].append(temp[0])
        results_dict['Factual '+dataset_type+' AUC'].append(temp[1])
    # wandb.log({"MSE"+'-Train-F': mse})
    print(dataset_type+' MSE: {:.4f}, AUC: {:.4f}'.format(mse, auc/args.num_task))

    return results_dict

def evaluate_counterfactual_model(model, X, X_train, X_test, Y_cf, y_cftrain, y_cftest, T, t_train, t_test, T_ind, 
                                  results_dict, args, logger, wandb, device):
    # to match train on SCP, splitting out val data
    X_train, X_val, y_cftrain, y_cfval, t_train, t_val = train_test_split(X_train, y_cftrain, t_train, test_size=args.val_size, random_state=42) #, stratify=t_train[:,-1]
    print((X_train.shape, t_train.shape))
    mse_train, auc_train, mse_test, auc_test, mse_all, auc_all = [], [], [], [], [], []
    
    # index to treatment combination dictionary
    T_combo = {v: k for k, v in T_ind.items()}
    for i in range(args.num_task):
        results_dict['CounterFactual All MSE-Y'+str(i)] = []
        results_dict['CounterFactual All AUC-Y'+str(i)] = []
        results_dict['CounterFactual Train MSE-Y'+str(i)] = []
        results_dict['CounterFactual Train AUC-Y'+str(i)] = []
        results_dict['CounterFactual Test MSE-Y'+str(i)] = []
        results_dict['CounterFactual Test AUC-Y'+str(i)] = []

    for i in range(len(T_combo)):
        combination = T_combo[i]
        # print('treatment:', i, combination)

        #  entire dataset
        T_temp = np.zeros_like(T)
        for j in range(T_temp.shape[1]):
            T_temp[:,j] = combination[j]
        out = model.predict(X, T_temp, device)
        mse = 0
        auc = 0
        for j in range(args.num_task):
            temp = calc_metrics(args, out[:,j], Y_cf[:,j,i], 'in', wandb, 1, logger, args.tasks[j])
            # print('All MSE: {:.4f}, AUC: {:.4f}'.format(temp[0], temp[1]))
            mse += temp[0]
            auc += temp[1]
            # save results
            results_dict['CounterFactual All MSE-Y'+str(j)].append(temp[0])
            results_dict['CounterFactual All AUC-Y'+str(j)].append(temp[1])
        # print('MSE: {:.4f}'.format(mse))
        # wandb.log({"MSE"+'-All-CF': mse})
        mse_all.append(mse)
        auc_all.append(auc/args.num_task)

        #  train dataset
        T_temp = np.zeros_like(t_train)
        for j in range(T_temp.shape[1]):
            T_temp[:,j] = combination[j]
        out = model.predict(X_train, T_temp, device)
        mse = 0
        auc = 0
        for j in range(args.num_task):
            temp = calc_metrics(args, out[:,j], y_cftrain[:,j,i], 'in', wandb, 1, logger, args.tasks[j])
            # print('Train MSE: {:.4f}, AUC: {:.4f}'.format(temp[0], temp[1]))
            mse += temp[0]
            auc += temp[1]
            # save results
            results_dict['CounterFactual Train MSE-Y'+str(j)].append(temp[0])
            results_dict['CounterFactual Train AUC-Y'+str(j)].append(temp[1])
        # wandb.log({"MSE"+'-Train-CF': mse})
        mse_train.append(mse)
        auc_train.append(auc/args.num_task)
        
        # test dataset
        T_temp = np.zeros_like(t_test)
        for j in range(T_temp.shape[1]):
            T_temp[:,j] = combination[j]
        out = model.predict(X_test, T_temp, device)
        mse = 0
        auc = 0
        for j in range(args.num_task):
            temp = calc_metrics(args, out[:,j], y_cftest[:,j,i], 'out', wandb, 1, logger, args.tasks[j])
            # print('Test MSE: {:.4f}, AUC: {:.4f}'.format(temp[0], temp[1]))
            mse += temp[0]
            auc += temp[1]
            # save results
            results_dict['CounterFactual Test MSE-Y'+str(j)].append(temp[0])
            results_dict['CounterFactual Test AUC-Y'+str(j)].append(temp[1])
        # wandb.log({"MSE"+'-Test-CF': mse})
        mse_test.append(mse)
        auc_test.append(auc/args.num_task)

    print('\nData sizes train:', (i+1)*(j+1)*X_train.shape[0], ' test:', (i+1)*(j+1)*X_test.shape[0], ' all:', (i+1)*(j+1)*X.shape[0])
    print('Train: MSE: {:.5f} , AUC: {:.5f}'.format(sum(mse_train)/len(mse_train), sum(auc_train)/len(auc_train)))
    print('Test:  MSE: {:.5f} , AUC: {:.5f}'.format(sum(mse_test)/len(mse_test), sum(auc_test)/len(auc_test)))
    print('All:   MSE: {:.5f} , AUC: {:.5f}'.format(sum(mse_all)/len(mse_all), sum(auc_all)/len(auc_all)))
    wandb.log({"All Train": sum(mse_train)/len(mse_train)})
    wandb.log({"All Test": sum(mse_test)/len(mse_test)})
    wandb.log({"All MSE": sum(mse_all)/len(mse_all)})

    results_dict['CounterFactual Train MSE']=sum(mse_train)/len(mse_train)
    results_dict['CounterFactual Test MSE']=sum(mse_test)/len(mse_test)
    results_dict['CounterFactual All MSE']=sum(mse_all)/len(mse_all)
    results_dict['CounterFactual Train AUC']=sum(auc_train)/len(auc_train)
    results_dict['CounterFactual Test AUC']=sum(auc_test)/len(auc_test)
    results_dict['CounterFactual All AUC']=sum(auc_all)/len(auc_all)

    return results_dict