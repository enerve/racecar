# '''
# Created on 1 Mar 2019
# 
# @author: enerve
# '''
# 
# import torch.nn as nn
# import torch.nn.functional as F
# 
# class Net(nn.Module):
#     
#     def __init__(self, layers):
#         super(Net, self).__init__()
#         
#         self.layers = layers
#         
#         self.f_list = nn.ModuleList()
#         for i, o in zip(layers, layers[1:]):
#             self.f_list.append(nn.Linear(i, o))
#         
#     def prefix(self):
#         return '%s' % self.layers
#         
#     def forward(self, x):
#         f = self.f_list[0]
#         z = f(x)
#         
#         for f in self.f_list[1:]:
#             a = F.relu(z)
#             z = f(a)
# 
#         return z

'''
Created on 1 Mar 2019

@author: enerve
'''

import torch.nn as nn
import torch.nn.functional as F

class Net(nn.Module):
    
    def __init__(self, num_inputs, num_hidden, num_output):
        super(Net, self).__init__()
        
        self.layers = [num_inputs, num_hidden, num_hidden, num_hidden, num_output]
        
        self.f1 = nn.Linear(num_inputs, num_hidden)
        self.f1_5 = nn.Linear(num_hidden, num_hidden)
        self.f1_75 = nn.Linear(num_hidden, num_hidden)
        self.f2 = nn.Linear(num_hidden, num_output)
        
    def prefix(self):
        return '%s' % self.layers
        
    def forward(self, x):
        z1 = self.f1(x)
        a1 = F.sigmoid(z1)
        z1_5 = self.f1_5(a1)
        a1_5 = F.sigmoid(z1_5)
        z1_75 = self.f1_5(a1_5)
        a1_75 = F.sigmoid(z1_75)
        z2 = self.f2(a1_75)
        return z2

