'''
Created on 1 Mar 2019

@author: enerve
'''

import torch.nn as nn
import torch.nn.functional as F

class Net(nn.Module):
    
    def __init__(self, num_inputs, num_hidden, num_output):
        super(Net, self).__init__()
        
        self.layers = [num_inputs, num_hidden, num_output]
        
        self.f1 = nn.Linear(num_inputs, num_hidden)
        self.f2 = nn.Linear(num_hidden, num_output)
        
    def prefix(self):
        return '%s' % self.layers
        
    def forward(self, x):
        z1 = self.f1(x)
        a1 = F.sigmoid(z1)
        z2 = self.f2(a1)
        return z2

