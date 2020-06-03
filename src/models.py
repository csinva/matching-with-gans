import torch
import torch.nn as nn
import torch.nn.functional as F
import FrEIA.framework as Ff
import FrEIA.modules as Fm
from FrEIA.framework import *

def get_INN(num_layers: int, input_size, hidden_size):
    '''Invertible neural network made of glow coupling blocks
    Params
    ------
    num_layers
        Really the number of glow coupling blocks
    hidden_size
        The hidden size for the FC in the glow blocks

    Note: output_size = input_size, should pad to make it work
    '''
    def subnet_fc(c_in, c_out):
        return nn.Sequential(nn.Linear(c_in, hidden_size),
                             nn.ReLU(),
                             nn.Linear(hidden_size,  c_out))

    # InputNode is just a node that is the input
    nodes = [Ff.InputNode(input_size, name='input')] 

    # Use a loop to produce a chain of coupling blocks
    for k in range(num_layers):
        nodes.append(Ff.Node(nodes[-1], # what does it take input from
                             Fm.GLOWCouplingBlock,
                             {'subnet_constructor':subnet_fc, 'clamp': 2.0},
                             name=F'coupling_{k}'))

    # OutputNode is just a node that yields the output
    nodes.append(Ff.OutputNode(nodes[-1], name='output')) 
    return Ff.ReversibleGraphNet(nodes)

        


    

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
    
    def last_lay(self):
        return self.fc[-1]
    
    
# make invertible net useing FreEIA https://github.com/VLL-HD/FrEIA