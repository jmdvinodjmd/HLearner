import torch
import torch.nn as nn
from torch.nn.parameter import Parameter
from torch.nn.utils.parametrizations import spectral_norm
import math

import src.utils as utils
from src.models.hn_utils import *


class HyperNetworkLayerwise(nn.Module):
    '''
        Layer-wise weight prediction for the target network.
            This also requires to keep separate embeddings for each layer of the target.
    '''
    def __init__(self, target_layers, target_in, hidden_layers=[100, 100], activations=[nn.ReLU, nn.ReLU, nn.ReLU],
                num_embs=2, emb_dim=16, dropout_rate=0.0, spect_norm=0):
        super(HyperNetworkLayerwise, self).__init__()
        # calculate total number of required params
        self.out_size = calc_params(target_in, target_layers)
        # calculate max number of parameters among target layers
        layer_weights = calc_layer_param_max(target_in, target_layers)
        self.target_layers = target_layers
        self.emb_dim = emb_dim
        self.dropout_rate = dropout_rate

        # create embedding
        # self.embedding_list = create_embeddings(num_embs * len(target_layers), emb_dim)
        self.embedding_list = nn.Embedding(num_embs * len(target_layers), emb_dim)

        # create hypernetwork layers
        self.layers = nn.ModuleList()
        # adding hidden layers
        for i in range(len(hidden_layers)):
            if i == 0:
                if spect_norm:
                    self.layers.append(spectral_norm(nn.Linear(emb_dim, hidden_layers[0])))
                else:
                    self.layers.append(nn.Linear(emb_dim, hidden_layers[0]))
                self.layers.append(activations[0]())
                self.layers.append(nn.Dropout(dropout_rate))
            else:
                if spect_norm:
                    self.layers.append(spectral_norm(nn.Linear(hidden_layers[i-1], hidden_layers[i])))
                else:
                    self.layers.append(nn.Linear(hidden_layers[i-1], hidden_layers[i]))
                self.layers.append(activations[i]())
                self.layers.append(nn.Dropout(dropout_rate))

        # adding the output layer of HN to predict weights equal to max among layers
        if spect_norm:
            self.layers.append(spectral_norm(nn.Linear(hidden_layers[-1], layer_weights)))
        else:
            self.layers.append(nn.Linear(hidden_layers[-1], layer_weights))

        # # initialise weights
        # self.apply(utils.init_weights)

        print('Created a hypernetwork with {:d} parameters for a target network with {:d} parameters.'.format(sum(p.numel() for p in self.parameters()), self.out_size))

    def forward(self, id, device='gpu'):
        id = torch.tensor([id]).to(device)

        weights = torch.Tensor([]).to(device)
        for i in range(len(self.target_layers)):
            # get the embedding first
            x = self.embedding_list(id * len(self.target_layers) + i).squeeze()
            # x = self.embedding_list[id * len(self.target_layers) + i]
            # generate weights for target layers one at time
            for layer in self.layers:
                x = layer(x)
            weights = torch.cat((weights, x))

        return weights


class HyperNetworkChunking(nn.Module):
    '''
        HyperNetworkChunking: This class generates target weights in chunks to reduce the complexity.
            This also requires to keep separate embeddings for each chunk of the target.
    '''
    def __init__(self, target_layers, target_in, num_chunks=1, hidden_layers=[100, 100], activations=[nn.ReLU, nn.ReLU, nn.ReLU],
                num_embs=2, emb_dim=16, dropout_rate=0.0, spect_norm=0):
        super(HyperNetworkChunking, self).__init__()

        self.out_size = calc_params(target_in, target_layers)
        self.emb_dim = emb_dim
        self.dropout_rate = dropout_rate
        self.num_chunks = num_chunks

        # create embedding
        self.embedding_list = nn.Embedding(num_embs * num_chunks, emb_dim)
        # self.embedding_list = create_embeddings(num_embs * num_chunks, emb_dim)

        # create network layers
        self.layers = nn.ModuleList()
        # adding hidden layers
        for i in range(len(hidden_layers)):
            if i == 0:
                if spect_norm:
                    self.layers.append(spectral_norm(nn.Linear(emb_dim, hidden_layers[0])))
                else:
                    self.layers.append(nn.Linear(emb_dim, hidden_layers[0]))
                self.layers.append(activations[0]())
                self.layers.append(nn.Dropout(dropout_rate))
            else:
                if spect_norm:
                    self.layers.append(spectral_norm(nn.Linear(hidden_layers[i-1], hidden_layers[i])))
                else:
                    self.layers.append(nn.Linear(hidden_layers[i-1], hidden_layers[i]))
                self.layers.append(activations[i]())
                self.layers.append(nn.Dropout(dropout_rate))

        # adding the output layer to predict weights
        head_output = math.ceil(self.out_size / float(num_chunks))

        if spect_norm:
            self.layers.append(spectral_norm(nn.Linear(hidden_layers[-1], head_output)))
        else:
            self.layers.append(nn.Linear(hidden_layers[-1], head_output))

        # # initialise weights
        # self.apply(utils.init_weights)

        print('Created a hypernetwork with {:d} parameters for a target network with {:d} parameters.'.format(sum(p.numel() for p in self.parameters()), self.out_size))

    def forward(self, id, device='gpu'):
        id = torch.tensor([id]).to(device)

        weights = torch.Tensor([]).to(device)
        for i in range(self.num_chunks):
            # get the embedding first
            x = self.embedding_list(id * self.num_chunks + i).squeeze()
            # x = self.embedding_list[id * self.num_chunks + i]
            # print(x.shape, 'xxxxxxxxxxxxxxxxxxxxxxxxx')
            # generate weights for each chunk one at time
            for layer in self.layers:
                x = layer(x)
            weights = torch.cat((weights, x))

        return weights


class HyperNetwork(nn.Module):
    '''
        This class generates target weights in one go so it the most complex.
            This requires to keep embeddings the target tasks only.

        @inproceedings{ha2017hypernetworks,
            title={HyperNetworks},
            author={David Ha and Andrew M. Dai and Quoc V. Le},
            booktitle={International Conference on Learning Representations},
            year={2017},
            url={https://openreview.net/forum?id=rkpACe1lx}
        }
    '''
    def __init__(self, target_layers, target_in, hidden_layers=[100, 100], activations=[nn.ReLU, nn.ReLU],
                num_task=1, num_cause=2, emb_dim_t=8, emb_dim_y=8, dropout_rate=0.0, spect_norm=0):
        super(HyperNetwork, self).__init__()

        self.out_size = calc_params(target_in, target_layers)
        self.dropout_rate = dropout_rate
        self.num_task = num_task

        # create embedding
        # self.embedding_list = create_embeddings(num_embs, emb_dim)
        # self.embedding_list = nn.Embedding(num_embs, emb_dim)
        # create treatment and task embeddings
        self.emb_t =  nn.Linear(num_cause, emb_dim_t, bias=False)
        hypernet_input_size = emb_dim_t
        if num_task>1:
            self.emb_y =  nn.Linear(num_task, emb_dim_y, bias=False)
            hypernet_input_size += emb_dim_y
        
        # create network layers
        self.layers = nn.ModuleList()
        # adding hidden layers
        for i in range(len(hidden_layers)):
            if i == 0:
                if spect_norm:
                    self.layers.append(spectral_norm(nn.Linear(hypernet_input_size, hidden_layers[0])))
                else:
                    self.layers.append(nn.Linear(hypernet_input_size, hidden_layers[0]))
                self.layers.append(activations[0]())
                self.layers.append(nn.Dropout(dropout_rate))
            else:
                if spect_norm:
                    self.layers.append(spectral_norm(nn.Linear(hidden_layers[i-1], hidden_layers[i])))
                else:
                    self.layers.append(nn.Linear(hidden_layers[i-1], hidden_layers[i]))
                self.layers.append(activations[i]())
                self.layers.append(nn.Dropout(dropout_rate))

        # adding the output layer to predict weights
        if spect_norm:
            self.layers.append(spectral_norm(nn.Linear(hidden_layers[-1], self.out_size)))
        else:
            self.layers.append(nn.Linear(hidden_layers[-1], self.out_size))
        
        # # initialise weights
        # self.apply(utils.init_weights)

        print('Created a hypernetwork with {:d} parameters for a target network with {:d} parameters.'.format(sum(p.numel() for p in self.parameters()), self.out_size))

    def forward(self, treatment, task, device='gpu'):
        # get the embedding first
        # x = self.embedding_list(torch.tensor(id, device=device))
        # # x = self.embedding_list[id]
        emb = self.emb_t(treatment)
        if self.num_task>1:
            e_y = self.emb_y(task)
            emb = torch.concat((emb, e_y), dim=1)
        
        x = emb
        for layer in self.layers:
            x = layer(x)

        return x


class HyperLearner(nn.Module):
    '''
    '''
    def __init__(self, hypernet, args, input_size, activations, 
                    target_layers, emb_dim, hn_drop_rate, spect_norm):
        
        super(HyperLearner, self).__init__()
        self.target_layers = target_layers
        self.activations = activations
        self.dropout_rate = args.drop_rate
        self.num_task = args.num_task
        self.tasks = args.tasks

        # store trained weights for N learners
        self.weights = [0]*args.num_task
        # create a hypernetwork
        if hypernet == 'layerwise':
            self.hypernetwork = HyperNetworkLayerwise(target_layers=target_layers, target_in=input_size, num_embs=self.N, 
                emb_dim=emb_dim, dropout_rate=hn_drop_rate, spect_norm=spect_norm)
        elif hypernet == 'chunking':
            self.hypernetwork = HyperNetworkChunking(num_chunks=args.num_chunks, target_layers=target_layers, target_in=input_size, 
                num_embs=self.N, emb_dim=emb_dim, dropout_rate=hn_drop_rate, spect_norm=spect_norm)
        elif hypernet == 'all':
            self.hypernetwork = HyperNetwork(target_layers=target_layers, target_in=input_size, num_task=args.num_task, 
                                             num_cause=args.num_cause, emb_dim_t=args.emb_dim_t, emb_dim_y=args.emb_dim_y,
                                        dropout_rate=hn_drop_rate, spect_norm=spect_norm)
        elif hypernet == 'split':
            self.hypernetwork = HyperNetworkSplitHead(target_layers=target_layers, target_in=input_size, num_embs=self.N, 
                emb_dim=emb_dim, dropout_rate=hn_drop_rate, spect_norm=spect_norm)
        
    def forward(self, X, comp_treat):
        device = next(self.parameters()).device
        # create functional MLP (learners) with weights from hypernet
        # use hypernetwork to generate weights for each task network
        for i in range(self.num_task):
            task = torch.zeros(self.num_task, device=X.device)
            task[i] = 1
            task = task.unsqueeze(0).repeat(X.shape[0], 1)
            self.weights[i] = self.hypernetwork(treatment=comp_treat, task=task, device=device)

        # create functional MLP and pass data to get output
        outputs_all = []
        for j in range(self.num_task):
            outputs = []
            activations = self.activations.copy()
            activations[-1][-1] = None if self.tasks[j]=='cont' else F.sigmoid
            for i in range(X.shape[0]):
                output = MLPFunctional(X[i,:], self.weights[j][i], in_size=X.shape[-1], layers=self.target_layers, 
                                activations=activations, dropout_rate=self.dropout_rate)
                outputs.append(torch.stack(output, dim=0))
            
            outputs_all.append(torch.stack(outputs, dim=0))
            
        return torch.stack(outputs_all, dim=1)
    
    def predict(self, X, comp_treat, device):
        for i in range(len(self.num_task)):
            task = torch.zeros(self.num_task, device=X.device)
            task[i] = 1
            task = task.unsqueeze(0).repeat(X.shape[0], 1)
            self.weights[i] = self.hypernetwork(treatment=comp_treat, task=task, device=device)
        # create functional MLP and pass data to get output
        outputs_all = []
        for j in range(len(self.num_task)):
            outputs = []
            activations = self.activations.copy()
            activations[-1][-1] = None if self.tasks[j]=='cont' else F.sigmoid
            for i in range(X.shape[0]):
                output = MLPFunctional(X[i,:], self.weights[j][i], in_size=X.shape[-1], layers=self.target_layers, 
                                activations=activations, dropout_rate=self.dropout_rate)
                outputs.append(torch.stack(output, dim=0))
            
            outputs_all.append(torch.stack(outputs, dim=0))
            
        return torch.stack(outputs_all, dim=1)
    

class HyperNLearner(nn.Module):
    '''
        This is a generic hypernet which can train N learners simultaneously.
    '''
    def __init__(self, N, hypernet, args, input_size, activations, 
                    target_layers, emb_dim, hn_drop_rate, spect_norm):
        
        super(HyperNLearner, self).__init__()
        self.target_layers = target_layers
        self.activations = activations
        self.dropout_rate = args.drop_rate
        self.N = N
        # store trained weights for N learners
        self.weights = [0]*self.N
        # create a hypernetwork
        if hypernet == 'layerwise':
            self.hypernetwork = HyperNetworkLayerwise(target_layers=target_layers, target_in=input_size, num_embs=self.N, 
                emb_dim=emb_dim, dropout_rate=hn_drop_rate, spect_norm=spect_norm)
        elif hypernet == 'chunking':
            self.hypernetwork = HyperNetworkChunking(num_chunks=args.num_chunks, target_layers=target_layers, target_in=input_size, 
                num_embs=self.N, emb_dim=emb_dim, dropout_rate=hn_drop_rate, spect_norm=spect_norm)
        elif hypernet == 'all':
            self.hypernetwork = HyperNetwork(target_layers=target_layers, target_in=input_size, num_embs=self.N, 
                emb_dim=emb_dim, dropout_rate=hn_drop_rate, spect_norm=spect_norm)
        elif hypernet == 'split':
            self.hypernetwork = HyperNetworkSplitHead(target_layers=target_layers, target_in=input_size, num_embs=self.N, 
                emb_dim=emb_dim, dropout_rate=hn_drop_rate, spect_norm=spect_norm)
        
    def forward(self, X, mask):
        device = next(self.parameters()).device
        outputs = []
        # create functional MLP (learners) with weights from hypernet
        for i in range(self.N):
            # use hypernetwork to generate weights
            self.weights[i] = self.hypernetwork(id=i, device=device)
            # create functional MLP and pass data to get output
            outputs.append(MLPFunctional(X[mask[i],:], self.weights[i], in_size=X.shape[-1], layers=self.target_layers, 
                                activations=self.activations[i], dropout_rate=self.dropout_rate))
            
        return outputs
    
    def predict(self, X, device):
        # X = torch.tensor(X).to(device).float()
        # print(self.weights)
        outputs = []
        # create functional MLP (learners) from weights learned during training
        for i in range(self.N):
            # create functional MLP and pass data to get output
            outputs.append(MLPFunctional(X, self.weights[i], in_size=X.shape[-1], layers=self.target_layers, 
                                activations=self.activations[i], dropout_rate=self.dropout_rate))                             

        return outputs

class HyperNetworkSplitHead(nn.Module):
    '''
    '''

    def __init__(self, target_layers, target_in, hidden_layers=[100, 100], activations=[nn.ReLU, nn.ReLU, nn.ReLU],
                num_embs=2, emb_dim=16, dropout_rate=0.0, spect_norm=0, num_heads=2):

        super(HyperNetworkSplitHead, self).__init__()
        self.num_heads = num_heads
        output_size = calc_params(target_in, target_layers)
        self.out_size = math.ceil(output_size/num_heads)
        self.emb_dim = emb_dim
        self.dropout_rate = dropout_rate

        # create embedding
        # self.embedding_list = create_embeddings(num_embs, emb_dim)
        self.embedding_list = nn.Embedding(num_embs, emb_dim)

        # create network layers
        self.layers = nn.ModuleList()
        # adding hidden layers
        for i in range(len(hidden_layers)-1):
            if i == 0:
                if spect_norm:
                    self.layers.append(spectral_norm(nn.Linear(emb_dim, hidden_layers[0])))
                else:
                    self.layers.append(nn.Linear(emb_dim, hidden_layers[0]))
                self.layers.append(activations[0]())
                self.layers.append(nn.Dropout(dropout_rate))
            else:
                if spect_norm:
                    self.layers.append(spectral_norm(nn.Linear(hidden_layers[i-1], hidden_layers[i])))
                else:
                    self.layers.append(nn.Linear(hidden_layers[i-1], hidden_layers[i]))
                self.layers.append(activations[i]())
                self.layers.append(nn.Dropout(dropout_rate))

        # create heads with output layers
        # head 1
        self.head1 = nn.ModuleList()
        if spect_norm:
            self.head1.append(spectral_norm(nn.Linear(hidden_layers[-2], int(hidden_layers[-1]/num_heads))))
        else:
            self.head1.append(nn.Linear(hidden_layers[-2], int(hidden_layers[-1]/num_heads)))
        self.head1.append(activations[-1]())
        self.head1.append(nn.Dropout(dropout_rate))
        if spect_norm:
            self.head1.append(spectral_norm(nn.Linear(int(hidden_layers[-1]/num_heads), self.out_size)))
        else:
            self.head1.append(nn.Linear(int(hidden_layers[-1]/num_heads), self.out_size))
        
        # head 2
        self.head2 = nn.ModuleList()
        if spect_norm:
            self.head2.append(spectral_norm(nn.Linear(hidden_layers[-2], int(hidden_layers[-1]/num_heads))))
        else:
            self.head2.append(nn.Linear(hidden_layers[-2], int(hidden_layers[-1]/num_heads)))
        self.head2.append(activations[-1]())
        self.head2.append(nn.Dropout(dropout_rate))
        if spect_norm:
            self.head2.append(spectral_norm(nn.Linear(int(hidden_layers[-1]/num_heads), self.out_size)))
        else:
            self.head2.append(nn.Linear(int(hidden_layers[-1]/num_heads), self.out_size))

        print('Created a hypernetwork with {:d} parameters for a target network with {:d} parameters.'.format(sum(p.numel() for p in self.parameters()), output_size))


    def forward(self, id, device='gpu'):
        # get the embedding first
        x = self.embedding_list(torch.tensor(id, device=device))
        # x = self.embedding_list[id]

        for layer in self.layers:
            x = layer(x)
        
        w1 = x
        for layer in self.head1:
            w1 = layer(w1)
        
        w2 = x
        for layer in self.head2:
            w2 = layer(w2)  

        return torch.cat((w1, w2), 0)


##############################
### Helper functions #########
def create_embeddings(num_embs, emb_dim):
    '''
        This function is used to create embeddings of given dimensions.
    '''
    embedding_list = nn.ParameterList()
    for i in range(num_embs):
        embedding_list.append(Parameter(torch.fmod(torch.randn(emb_dim).cuda(), 2)))

    return embedding_list

def calc_params(in_size, target_layers):
    '''
    This function calculates number of parameters required in the target network
    '''
    total = 0
    # calculate params in the trunk
    for i in range(len(target_layers[0])):
        if i==0:
            total += target_layers[0][i] + target_layers[0][i] * in_size
        else:
            total += target_layers[0][i] + target_layers[0][i] * target_layers[0][i-1]

    # calculate params in task-heads
    for j in range(len(target_layers[1:])):
        for i in range(len(target_layers[j+1])):
            if i==0:
                total += target_layers[j+1][i] + target_layers[j+1][i] * target_layers[0][-1]
            else:
                total += target_layers[j+1][i] + target_layers[j+1][i] * target_layers[j+1][i-1]

    return total

def calc_layer_param_max(target_in, target_layers):
    '''
        This function calculates max number of params among different target layers.
    '''
    max_w = 0
    for i in range(len(target_layers)):
        if i==0:
            temp = target_in * target_layers[i] + target_layers[i]
        else:
            temp = target_layers[i-1] * target_layers[i] + target_layers[i]

        if max_w < temp:
            max_w = temp

    return max_w
##############################
