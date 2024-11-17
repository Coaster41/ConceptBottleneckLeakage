import torch.nn as nn
import torch
import time
import os
import matplotlib.pyplot as plt

from train import load_data, train, eval
from models import *
import json
import argparse
from loss import leakage_loss

import git

repo = git.Repo(search_parent_directories=True)
sha = repo.head.object.hexsha
sha = repo.git.rev_parse(sha, short=8)

device = 'cuda' if torch.cuda.is_available() else 'cpu'
print(device)

parser = argparse.ArgumentParser()

parser.add_argument('-c', "--config", required=True)    

args = parser.parse_args()

with open(args.config, 'r') as file:
    config = json.load(file)

# Load from config
data_fn = config.get("data_loc")
X_headers = config.get("x_headers")
C_headers = config.get("c_headers")
Y_headers = config.get("y_headers")
batch_size = config.get("batch_size", 1024)
Cy_norm = config.get("normalize", True)
latents = config.get("n_latents", 0)
label_loss_str = config.get("y_criterion", "bce")
loss_dict = {"bce": nn.BCELoss(), "bce_logits": nn.BCEWithLogitsLoss(), "mse": nn.MSELoss(),
             "leakage": leakage_loss, "cross_entropy": nn.CrossEntropyLoss()}
label_criterion = loss_dict.get(label_loss_str, nn.BCELoss())
label_weight = config.get("y_weight", 1)
concept_loss_str = config.get("c_criterion", "mse")
if isinstance(concept_loss_str, list):
    concept_criterion = []
    for criterion_str in concept_loss_str:
        concept_criterion.append(loss_dict.get(criterion_str, nn.MSELoss()))
else:
    concept_criterion = loss_dict.get(concept_loss_str, nn.MSELoss())
concept_weight = config.get("c_weight", 0.5)
latent_loss_str = config.get("l_criterion", None)
if not latent_loss_str:
    latent_criterion = None
else:
    latent_criterion = loss_dict.get(latent_loss_str, leakage_loss)
latent_weight = config.get("l_weight", 0)
learning_rate = config.get("lr", 0.001)
optimizer_str = config.get("optimizer", "adam")
optimizer_dict = {"adam": torch.optim.Adam, "sgd": torch.optim.SGD, "nadam": torch.optim.NAdam}
optimizer_base = optimizer_dict.get(optimizer_str, torch.optim.Adam)

x_to_c_params = config.get("x_to_c_model", dict())
xc_depth = x_to_c_params.get('depth', 4)
xc_width = x_to_c_params.get('width', 32)
xc_use_sigmoid = x_to_c_params.get('use_sigmoid', False)
xc_use_relu = x_to_c_params.get('use_relu', True if not xc_use_sigmoid else False)
xc_final_activation_str = x_to_c_params.get('final_activation', True)
activation_dict = {"relu": nn.ReLU(), "sigmoid": nn.Sigmoid(), "tanh": nn.Tanh(), "lrelu": nn.LeakyReLU()}
if isinstance(xc_final_activation_str, str):
    xc_final_activation = activation_dict.get(xc_final_activation_str, False)
else:
    xc_final_activation = xc_final_activation_str

c_to_y_params = config.get("c_to_y_model", dict())
cy_depth = c_to_y_params.get('depth', 3)
cy_width = c_to_y_params.get('width', 16)
cy_use_sigmoid = c_to_y_params.get('use_sigmoid', False)
cy_use_relu = c_to_y_params.get('use_relu', True if not cy_use_sigmoid else False)
cy_final_activation_str = c_to_y_params.get('final_activation', "sigmoid")
if isinstance(cy_final_activation_str, str):
    cy_final_activation = activation_dict.get(cy_final_activation_str, None)
else:
    cy_final_activation = cy_final_activation_str

checkpoint_path = config.get("checkpoint_path", "checkpoints/")


# Setup Model

train_loader, test_loader, Cy_max = load_data(X_headers, C_headers, Y_headers, data_fn, Cy_norm=Cy_norm, batch_size=batch_size)

x_to_c_model = XtoCModel(len(X_headers), len(C_headers), latents, depth=xc_depth, width=xc_width,
                         use_relu=xc_use_relu, use_sigmoid=xc_use_sigmoid, final_activation=xc_final_activation).to(device)
c_to_y_model = CtoYModel(len(C_headers)+latents, len(Y_headers), depth=cy_depth, width=cy_width,
                         use_relu=cy_use_relu, use_sigmoid=cy_use_sigmoid, final_activation=cy_final_activation).to(device)
model = FullModel(x_to_c_model, c_to_y_model).to(device)

optimizer = optimizer_base(model.parameters(), lr=learning_rate)

# Train Model

losses, accuracies = train(model, train_loader, len(C_headers), optimizer, y_criterion=label_criterion,
               concept_criterion=concept_criterion, y_weight=label_weight, 
               concept_weight=concept_weight, device=device)

if not os.path.exists(checkpoint_path+"joint_cbm_"+str(sha)):
    os.makedirs(checkpoint_path+"joint_cbm_"+str(sha))
torch.save(model.state_dict(), checkpoint_path+'joint_cbm_'+str(sha)+'/model.pkl')
json_obj = json.dumps(config, indent=4)
with open(checkpoint_path+'joint_cbm_'+str(sha)+'/config.yaml', 'w') as outfile:
	outfile.write(json_obj)
print(f'model saved to: {checkpoint_path}joint_cbm_{str(sha)}')

test_loss, test_accuracy = eval(model, test_loader, len(C_headers), y_criterion=label_criterion,
               concept_criterion=concept_criterion, y_weight=label_weight, 
               concept_weight=concept_weight, device=device)

fig, ax1 = plt.subplots()
color = 'tab:red'
ax1.set_xlabel('Epoch')
ax1.set_ylabel('Loss', color=color)
ax1_train_loss = ax1.plot(range(len(losses)), losses, color=color, label="Training Loss")
ax1_test_loss = ax1.axhline(y = test_loss, color = color, linestyle = 'dashed', label="Test Loss") 
ax1.tick_params(axis='y', labelcolor=color)

ax2 = ax1.twinx()  # instantiate a second Axes that shares the same x-axis

color = 'tab:blue'
ax2.set_ylabel('Accuracy', color=color)  # we already handled the x-label with ax1
ax2_train_accuracy = ax2.plot(range(len(accuracies)), accuracies, color=color, label="Train Accuracy")
ax2_test_accuracy = ax2.axhline(y = test_accuracy, color = color, linestyle = 'dashed', label="Test Accuracy") 
ax2.tick_params(axis='y', labelcolor=color)

plots = ax1_train_loss+[ax1_test_loss]+ax2_train_accuracy+[ax2_test_accuracy]
labs = [l.get_label() for l in plots]
ax1.legend(plots, labs, loc='right')

fig.tight_layout()  # otherwise the right y-label is slightly clipped
plt.savefig(checkpoint_path+'joint_cbm_'+str(sha)+"/losses.png") 

print('Test Loss:', test_loss)
