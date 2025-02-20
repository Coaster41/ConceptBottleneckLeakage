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
    def __init__(self, x_to_c_model, c_to_y_model, device='cpu', concepts=None, latents=None):
        super(FullModel, self).__init__()
        self.x_to_c_model = x_to_c_model
        self.c_to_y_model = c_to_y_model
        if concepts is not None:
            self.c_num = concepts
        else:
            self.c_num = self.x_to_c_model.concepts
        if latents is not None:
            self.l_num = latents
        else:
            self.l_num = self.x_to_c_model.latents
        self.device = device
        self.ixs = torch.arange(self.c_num+self.l_num, dtype=torch.int64).to(self.device)
    
    def forward(self, x, use_latents=True, hard_cbm=False):
        c_out = self.x_to_c_model(x)
        if not use_latents and self.l_num > 0:
            if hard_cbm:
                c_out_hard = torch.where(self.ixs[None, :] >= self.c_num, torch.tensor(0.).to(self.device), c_out + torch.round(c_out).detach() - c_out.detach())
            c_out = torch.where(self.ixs[None, :] >= self.c_num, torch.tensor(0.).to(self.device), c_out)
        elif hard_cbm:
            c_out_hard = torch.where(self.ixs[None, :] >= self.c_num, c_out, c_out + torch.round(c_out).detach() - c_out.detach())
        y_out = self.c_to_y_model(c_out if not hard_cbm else c_out_hard)
        return c_out, y_out
    
    def forward_concepts(self, c, l, hard_cbm=False):
        if hard_cbm:
            c = c + torch.round(c).detach() - c.detach()
        cl = torch.cat((c, l), dim=1)
        return self.c_to_y_model(cl)

    
class ThreePartModel(torch.nn.Module):
    def __init__(self, x_to_c_model, x_to_l_model, cl_to_y_model, device='cpu'):
        super(ThreePartModel, self).__init__()
        self.x_to_c_model = x_to_c_model
        self.x_to_l_model = x_to_l_model
        self.cl_to_y_model = cl_to_y_model
        self.c_num = self.x_to_c_model.concepts
        self.l_num = self.x_to_c_model.latents
        self.device = device

    def freeze_x_to_c(self, freeze=True):
        for param in self.x_to_c_model.parameters():
            param.requires_grad = not freeze
    
    def forward(self, x, use_latents=True, hard_cbm=False):
        c_out = self.x_to_c_model(x)
        if self.x_to_l_model:
            if use_latents and self.l_num > 0:
                l_out = self.x_to_l_model(x)
                if hard_cbm:
                    c_out_hard = torch.cat([c_out + torch.round(c_out).detach() - c_out.detach(), l_out], axis=1)
                
                c_out = torch.cat([c_out, l_out], axis=1)
            else:
                c_out = torch.cat([c_out, torch.zeros((c_out.shape[0],self.l_num), device=self.device)])
                if hard_cbm:
                    c_out_hard = c_out + torch.round(c_out).detach() - c_out.detach()
        y_out = self.cl_to_y_model(c_out if not hard_cbm else c_out_hard)
        return c_out, y_out

    def forward_concepts(self, c, l, hard_cbm=False):
        if hard_cbm:
            c = c + torch.round(c).detach() - c.detach()
        cl = torch.cat((c, l), dim=1)
        return self.cl_to_y_model(cl)
    
class Cifar10CnnModel(nn.Module):
    def __init__(self, concepts, latents, final_activation=None):
        super(Cifar10CnnModel, self).__init__()
        if final_activation == None:
            final_activation = nn.Sigmoid()
        self.concepts = concepts
        self.latents = latents
        self.network = nn.Sequential(
            nn.Conv2d(3, 32, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.Conv2d(32, 64, kernel_size=3, stride=1, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(2, 2), # output: 64 x 16 x 16
            nn.Dropout2d(p=0.2),

            nn.Conv2d(64, 128, kernel_size=3, stride=1, padding=1),
            nn.ReLU(),
            nn.Conv2d(128, 128, kernel_size=3, stride=1, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(2, 2), # output: 128 x 8 x 8
            nn.Dropout2d(p=0.2),

            nn.Conv2d(128, 256, kernel_size=3, stride=1, padding=1),
            nn.ReLU(),
            nn.Conv2d(256, 256, kernel_size=3, stride=1, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(2, 2), # output: 256 x 4 x 4
            nn.Dropout2d(p=0.2),

            nn.Flatten(), 
            nn.Linear(256*4*4, concepts+latents),
            final_activation)
        
    def forward(self, xb):
        return self.network(xb)