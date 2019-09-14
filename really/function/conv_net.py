'''
Created on 2 May 2019

@author: enerve
'''

import torch.nn as nn
import torch


class AllSequential(nn.Module):
    
    def __init__(self, models_dict):
        super().__init__()
        for name, module in models_dict.items():
            self.add_module(name, module)
        self.names = list(models_dict.keys())
        
    def get_names(self):
        return self.names
    
    def forward(self, x):
        act_list = []
        for module in self._modules.values():
            x = module(x)
            act_list.append(x)
        return act_list

class Flatten(nn.Module):
    def forward(self, x):
        #N, C, H, W = x.size() # read in N, C, H, W
        #return x.view(N, -1)
        #x = x.view(-1)
        #return x.unsqueeze(0)
        return torch.flatten(x, start_dim=1)

# class ExpandRange(nn.Module):
#     def forward(self, x):
#         return 2 * x - 1