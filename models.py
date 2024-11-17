import torch
import torch.nn as nn


class XtoCModel(nn.Module):
    def __init__(self, inputs, concepts, latents=0, depth=4, width=32, 
                 use_relu=True, use_sigmoid=False, activation=None, final_activation=True):
        super(XtoCModel, self).__init__()
        if use_relu:
            self.activation_func = nn.ReLU()
        elif use_sigmoid:
            self.activation_func = nn.Sigmoid()
        else:
            assert activation is not None
            self.activation_func = activation
        self.linears = []
        prev_width = inputs
        for i in range(depth):
            if i == depth - 1:
                self.linears.append(nn.Linear(prev_width, concepts+latents))
            else:
                if isinstance(width, list):
                    self.linears.append(nn.Linear(prev_width, width[i]))
                    prev_width = width[i]
                else:
                    self.linears.append(nn.Linear(prev_width, width))
                    prev_width = width
        self.linears = nn.ModuleList(self.linears)
        self.final_activation = final_activation


    def forward(self, x):
        for layer_num, layer in enumerate(self.linears):
            x = layer(x)
            if layer_num == len(self.linears)-1:
                if isinstance(self.final_activation, nn.Module):
                    x = self.final_activation(x)
                elif self.final_activation:
                    x = self.activation_func(x)
            else:
                x = self.activation_func(x)
        return x
        
class CtoYModel(nn.Module):
    def __init__(self, concepts, outputs, depth=3, width=16, 
                 use_relu=True, use_sigmoid=False, activation=None, final_activation=None):
        super(CtoYModel, self).__init__()
        if use_relu:
            self.activation_func = nn.ReLU()
        elif use_sigmoid:
            self.activation_func = nn.Sigmoid()
        else:
            assert activation is not None
            self.activation_func = activation
        self.linears = []
        prev_width = concepts
        for i in range(depth):
            if i == depth - 1:
                self.linears.append(nn.Linear(prev_width, outputs))
            else:
                if isinstance(width, list):
                    self.linears.append(nn.Linear(prev_width, width[i]))
                    prev_width = width[i]
                else:
                    self.linears.append(nn.Linear(prev_width, width))
                    prev_width = width
        self.linears = nn.ModuleList(self.linears)

        self.final_activation = final_activation

    def forward(self, x):
        for layer_num, layer in enumerate(self.linears):
            x = layer(x)
            if layer_num == len(self.linears)-1:
                if self.final_activation:
                    x = self.final_activation(x)
                else:
                    x = self.activation_func(x)
        return x
        
class FullModel(torch.nn.Module):
    def __init__(self, x_to_c_model, c_to_y_model):
        super(FullModel, self).__init__()
        self.x_to_c_model = x_to_c_model
        self.c_to_y_model = c_to_y_model
    
    def forward(self, x):
        c_out = self.x_to_c_model(x)
        y_out = self.c_to_y_model(c_out)
        return c_out, y_out