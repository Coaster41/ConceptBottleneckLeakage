import torch
from torch import nn
device = 'cuda' if torch.cuda.is_available() else 'cpu'

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
    nn.Linear(32, 32),
    nn.Sigmoid(),
    nn.Linear(32, num_c)
)

xc_to_y = nn.Sequential(
    nn.Linear(num_x-num_xc+num_c, 32),
    nn.Sigmoid(),
    nn.Linear(32, num_y)
)

xc_activation = BinarySigmoid()
cy_activation = nn.Softmax(dim=1)

init_weights(x_to_c)
init_weights(xc_to_y)

def gen_x(num_x, batch_size):
    return torch.rand((batch_size, num_x))

x = gen_x(num_x, num_samples)
c_logits = x_to_c(x[:,:num_xc])
mean_c_logits = torch.mean(c_logits, dim=0)
x_to_c[-1].bias -= mean_c_logits
c_logits = x_to_c(x[:,:num_xc])
c = xc_activation(c_logits)
y_logits = xc_to_y(torch.concat((x[:,num_xc:],c), dim=1))
mean_y_logits = torch.mean(y_logits, dim=0)
xc_to_y[-1].bias -= mean_y_logits
y_logits = xc_to_y(torch.concat((x[:,num_xc:],c), dim=1))
y = cy_activation(y_logits / torch.std(y_logits, dim=0))
y_argmax = torch.argmax(y, dim=1)
# print(torch.unique(y_argmax, return_counts=True))

x_to_new_x = nn.Sequential(
    nn.Linear(num_x, num_x)
)
init_weights(x_to_new_x)
new_x = x_to_new_x(x)

xyc = torch.concat([x,y,c], dim=1)
fn = '../../data/synthetic/xyc_easy.pt'
torch.save(xyc, fn)
xyc_hard = torch.concat([new_x,y,c], dim=1)
fn = '../../data/synthetic/xyc_hard.pt'
torch.save(xyc_hard, fn)