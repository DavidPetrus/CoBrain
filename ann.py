import numpy as np
import torch


class ANN(torch.nn.Module):
    def __init__(self, num_layers, num_filters):
        super(ANN, self).__init__()
        self.num_layers = num_layers
        self.num_filters = num_filters

        self.net_dict = {}
        self.net_dict['conv_1'] = torch.nn.Conv2d(1,self.num_filters,kernel_size=3,stride=2)
        for l in range(1,self.num_layers-1):
            self.net_dict['conv_'+str(l+1)] = torch.nn.Conv2d(self.num_filters,self.num_filters,kernel_size=3,stride=2)
        self.net_dict['fc_'+str(self.num_layers)] = torch.nn.Linear(self.num_filters,1,bias=False)
        self.net = torch.nn.ModuleDict(self.net_dict)
        self.relu = torch.nn.ReLU()
        self.sigmoid = torch.nn.Sigmoid()

    def forward(self, stimulus):
        activ = stimulus
        for key in self.net.keys():
            if key == 'fc_'+str(self.num_layers):
                break
            activ = self.relu(self.net[key](activ))

        activ = activ.mean([2,3])
        pred = self.sigmoid(self.net['fc_'+str(self.num_layers)](activ))

        return pred
