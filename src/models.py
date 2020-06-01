import torch
import torch.nn as nn
import torch.nn.functional as F

'''MLP which takes arguments for number of layers, sizes 
'''
class LinearNet(nn.Module):
    def __init__(self, num_layers: int, input_size, output_size, hidden_size=None):
        '''
        Params
        ------
        num_layers
            Number of weight matrices (so 1 is linear regression)
        '''
        super(LinearNet, self).__init__()
        self.input_size = input_size
        self.output_size = output_size
        
        if num_layers == 1:
            self.fc = nn.ModuleList([nn.Linear(input_size, self.output_size)])
        else:
            self.fc = nn.ModuleList([nn.Linear(input_size, hidden_size)])
            self.fc.extend([nn.Linear(hidden_size, hidden_size) for i in range(num_layers - 2)])
            self.fc.append(nn.Linear(hidden_size, self.output_size))
            
    # doesn't use last layer
    def features(self, x):
        y = x.view(-1, self.input_size)
        for i in range(len(self.fc) - 1):
            y = F.relu(self.fc[i](y))
        return y
        
    def forward(self, x):
        return self.fc[-1](self.features(x)) # last layer has no relu

    def forward_all(self, x):
        y = x.view(-1, self.input_size)
        out = {}
        for i in range(len(self.fc) - 1):
            y = self.fc[i](y)
            out['fc.' + str(i)] = y.data.clone() #deepcopy(y)
            y = F.relu(y)
        out['fc.' + str(len(self.fc) - 1)] = self.fc[-1](y).clone() # deepcopy(self.fc[-1](y))
        return out
    
    def last_lay(self): return self.fc[-1]