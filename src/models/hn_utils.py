'''
This file contains utilities used in the development of hypernetworks.

@author:
    Vinod Kumar Chauhan
    Institute of Biomedical Engineering
    University of Oxford, UK
'''
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np



def MLPFunctional(inputs, weights, in_size=8, layers=[[100, 100],[100, 1],[100, 1]], activations=[[F.elu, F.elu],[F.elu, None],[F.elu, None]], 
                  dropout_rate=0.0):
    '''
        Generic function to create MLP with given parameters.
    '''
    w = shapeWeights(weights, in_size, layers)
    x = inputs

    # processing through common trunk
    for i in range(len(layers[0])):
        x = nn.functional.linear(x, weight=w[0][i][0], bias=w[0][i][1])
        if activations[0][i] is not None:
            x = activations[0][i](x)
        if i>0:
            x = nn.functional.dropout(x, dropout_rate)

    # processing through the task-specific layers
    out = []
    for i in range(len(layers[1:])):
        temp = x
        for j in range(len(layers[i+1])):
            temp = nn.functional.linear(temp, weight=w[i+1][j][0], bias=w[i+1][j][1])
            if activations[i+1][j] is not None:
                temp = activations[i+1][j](temp)
        out.append(temp)
    
    return out


def shapeWeights(weights, in_size, layers):
    # print(weights.shape)
    w = []
    t = 0
    for j in range(len(layers)):
        w_l = []
        for i in range(len(layers[j])):
            if (j==0) and (i==0):
                size = layers[j][i] + layers[j][i] * in_size
                wt = weights[t:t+size]
                w_l.append([wt[:-layers[j][i]].view(layers[j][i], in_size), wt[-layers[j][i]]])
            elif (j>0) and (i==0):
                size = layers[j][i] + layers[j][i] * layers[0][-1]
                wt = weights[t:t+size]
                w_l.append([wt[:-layers[j][i]].view(layers[j][i], layers[0][-1]), wt[-layers[j][i]]])
            else:
                size = layers[j][i] + layers[j][i] * layers[j][i-1]
                wt = weights[t:t+size]
                w_l.append([wt[:-layers[j][i]].view(layers[j][i], layers[j][i-1]), wt[-layers[j][i]]])        
            t += size
        w.append(w_l)
            
    return w