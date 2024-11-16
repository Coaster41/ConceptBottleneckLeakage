import torch.nn as nn
import torch
import time
import os
import matplotlib.pyplot as plt

from train import load_data, train, eval
from models import *

import git

repo = git.Repo(search_parent_directories=True)
sha = repo.head.object.hexsha
sha = repo.git.rev_parse(sha, short=8)

device = 'cuda' if torch.cuda.is_available() else 'cpu'
print(device)

data_fn = "data/college_acceptance_100001_gaussian.csv"
X_headers = ['age', 'sex', 'white', 'asian', 'black', 'hispanic', 'otherRace', 'STEM', 'percentile', 'collegeLevel']
C_headers = ['gpa', 'SAT', 'SATWriting', 'APTests', 'APScores', 'essays', 'extracurriculars', 'collegeLevel']
Y_headers = ['accepted']

train_loader, test_loader, Cy_max = load_data(X_headers, C_headers, Y_headers, data_fn, Cy_norm=True, batch_size=1024)

latents = 0

x_to_c_model = XtoCModel(len(X_headers), len(C_headers), latents, final_activation=nn.ReLU()).to(device)
c_to_y_model = CtoYModel(len(C_headers)+latents, len(Y_headers), final_activation=nn.Sigmoid()).to(device)
model = FullModel(x_to_c_model, c_to_y_model).to(device)

label_loss = nn.BCELoss()
learning_rate = 0.001
optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)
label_loss_weight = 1
# concept_loss_weight = 0.2
# concept_criterion = []
# for _ in range(len(C_headers)):
#     concept_criterion.append(nn.MSELoss())
concept_loss_weight = 0.5
concept_criterion = nn.MSELoss()

losses, accuracies = train(model, train_loader, len(C_headers), optimizer, y_criterion=label_loss,
               concept_criterion=concept_criterion, y_weight=label_loss_weight, 
               concept_weight=concept_loss_weight, device=device)

if not os.path.exists("checkpoints/joint_cbm_"+str(sha)):
    os.makedirs("checkpoints/joint_cbm_"+str(sha))
torch.save(model.state_dict(), 'checkpoints/joint_cbm_'+str(sha)+'/model.pkl')
print(f'model saved to: checkpoints/joint_cbm_{str(sha)}')

test_loss, test_accuracy = eval(model, test_loader, len(C_headers), y_criterion=label_loss,
               concept_criterion=concept_criterion, y_weight=label_loss_weight, 
               concept_weight=concept_loss_weight, device=device)

fig, ax1 = plt.subplots()
color = 'tab:red'
ax1.set_xlabel('Epoch')
ax1.set_ylabel('Loss', color=color)
ax1.plot(range(len(losses)), losses, color=color)
ax1.axhline(y = test_loss, color = color, linestyle = 'dashed') 
ax1.tick_params(axis='y', labelcolor=color)
ax1.legend(["Training Loss", "Test Loss"])

ax2 = ax1.twinx()  # instantiate a second Axes that shares the same x-axis

color = 'tab:blue'
ax2.set_ylabel('Accuracy', color=color)  # we already handled the x-label with ax1
ax2.plot(range(len(accuracies)), accuracies, color=color)
ax2.axhline(y = test_accuracy, color = color, linestyle = 'dashed') 
ax2.tick_params(axis='y', labelcolor=color)
ax2.legend(["Training Accuracy", "Test Accuracy"])

fig.tight_layout()  # otherwise the right y-label is slightly clipped
plt.savefig('checkpoints/joint_cbm_'+str(sha)+"/losses.png") 

print('Test Loss:', test_loss)
