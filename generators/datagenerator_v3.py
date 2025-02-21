import torch
from torch import nn
import numpy as np
from sklearn.cluster import KMeans
from sklearn.decomposition import PCA
from sklearn.mixture import GaussianMixture

num_x = 64
num_c = 16
num_y = 10
k = 0.875 # ratio of x's to use to calculate concepts
num_xc = int(k * num_x)
num_samples = 10**5


class BinarySigmoid(nn.Module):
    def __init__(self):
        super(BinarySigmoid, self).__init__()

    def __repr__(self):
        return 'BinarySigmoid()'

    def forward(self, x):
        x = torch.nn.functional.sigmoid(x)
        return x + torch.round(x).detach() - x.detach()
    
def init_weights(model):
    for m in model.modules():
        if isinstance(m, nn.Linear):
            nn.init.xavier_normal_(m.weight)
            nn.init.normal_(m.bias)
            m.requires_grad_(False)

x_to_c = nn.Sequential(
    nn.Linear(num_xc, 32),
    nn.Sigmoid(),
    nn.Linear(32, num_c)
)

xc_activation = BinarySigmoid()

init_weights(x_to_c)

x  = torch.normal(0.5,0.63,(num_samples, num_x))

c_logits = x_to_c(x[:,:num_xc])
mean_c_logits = torch.mean(c_logits, dim=0)
x_to_c[-1].bias -= mean_c_logits
c_logits = x_to_c(x[:,:num_xc])
c = xc_activation(c_logits)

kmeans = KMeans(n_clusters=num_y).fit(torch.concat((x[:,num_xc:],c), dim=1))
y_argmax = torch.tensor(kmeans.labels_).long()
print(torch.unique(y_argmax, return_counts=True))

x_to_new_x = nn.Sequential(
    nn.Linear(num_x, num_x)
)
init_weights(x_to_new_x)
new_x = x_to_new_x(x)

y_argmax = y_argmax.unsqueeze(dim=1)

xyc = torch.concat([x,y_argmax,c], dim=1)
fn = '../data/synthetic/xyc_v3_easy.pt'
torch.save(xyc, fn)
xyc_hard = torch.concat([new_x,y_argmax,c], dim=1)
fn = '../data/synthetic/xyc_v3_hard.pt'
torch.save(xyc_hard, fn)