import math
import torch
import torch.nn as nn
import torch.nn.init as init 
import torch.nn.functional as F

def call_bn(bn, x):
    return bn(x)


class FC(nn.Module):
    def __init__(self, num_input, num_output, dropout_rate, hidden_dim):
        self.dropout_rate = dropout_rate
        super(FC,self).__init__()
        self.n_layers = len(hidden_dim)
        self.l1 = nn.Linear(num_input,hidden_dim[0])
        self.l2 = nn.Linear(hidden_dim[0],hidden_dim[1])
        self.lo = nn.Linear(hidden_dim[1],num_output)
    def forward(self,x):
        h = x 
        h = self.l1(h)
        h = F.relu(h)
        h = F.dropout(h, p=self.dropout_rate)

        h = self.l2(h)
        h = F.relu(h)
        h = F.dropout(h, p=self.dropout_rate)

        feat = h.view(h.size(0), -1)
        x = self.lo(feat)
        return x
        

def fcmodel(**kwargs):
    model = FC(**kwargs)
    return model 

