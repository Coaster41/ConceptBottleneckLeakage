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
        self.concepts = concepts
        self.latents = latents

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
    def __init__(self, x_to_c_model, c_to_y_model, device='cpu'):
        super(FullModel, self).__init__()
        self.x_to_c_model = x_to_c_model
        self.c_to_y_model = c_to_y_model
        self.c_num = self.x_to_c_model.concepts
        self.l_num = self.x_to_c_model.concepts
        self.device = device
        self.ixs = torch.arange(self.c_num+self.l_num, dtype=torch.int64).to(self.device)
    
    def forward(self, x, use_latents=True):
        c_out = self.x_to_c_model(x)
        if not use_latents:
            c_out = torch.where(self.ixs[None, :] >= self.c_num, torch.tensor(0.).to(self.device), c_out)
        y_out = self.c_to_y_model(c_out)
        return c_out, y_out
    
class ThreePartModel(torch.nn.Module):
    def __init__(self, x_to_c_model, x_to_l_model, cl_to_y_model, device='cpu'):
        super(ThreePartModel, self).__init__()
        self.x_to_c_model = x_to_c_model
        self.x_to_l_model = x_to_l_model
        self.cl_to_y_model = cl_to_y_model
        self.c_num = self.x_to_c_model.concepts
        self.l_num = self.x_to_c_model.concepts
        self.device = device
        self.ixs = torch.arange(self.c_num+self.l_num, dtype=torch.int64).to(self.device)

    def freeze_x_to_c(self, freeze=True):
        for param in self.x_to_c_model.parameters():
            param.requires_grad = not freeze
    
    def forward(self, x, use_latents=True):
        c_out = self.x_to_c_model(x)
        if self.x_to_l_model:
            l_out = self.x_to_l_model(x)
            c_out = torch.cat([c_out, l_out], axis=1)
        if not use_latents:
            c_out = torch.where(self.ixs[None, :] >= self.c_num, torch.tensor(0.).to(self.device), c_out)
        y_out = self.cl_to_y_model(c_out)
        return c_out, y_out
