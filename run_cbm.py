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

x_to_c_model = XtoCModel(len(X_headers), len(C_headers), latents, final_activation=nn.Sigmoid).to(device)
c_to_y_model = CtoYModel(len(C_headers)+latents, len(Y_headers)).to(device)
model = FullModel(x_to_c_model, c_to_y_model).to(device)

label_loss = nn.BCELoss()
learning_rate = 0.001
optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)
label_loss_weight = 1
concept_loss_weight = 0.5
concept_criterion = nn.MSELoss()

losses = train(model, train_loader, len(C_headers), optimizer, y_criterion=label_loss,
               concept_criterion=concept_criterion, y_weight=label_loss_weight, 
               concept_weight=concept_loss_weight, device=device)

if not os.path.exists("checkpoints/joint_cbm_"+str(sha)):
    os.makedirs("checkpoints/joint_cbm_"+str(sha))
torch.save(model.state_dict(), 'checkpoints/joint_cbm_'+str(sha)+'/model.pkl')
print(f'model saved to: checkpoints/joint_cbm_{str(sha)}')

test_loss = eval(model, test_loader, len(C_headers), y_criterion=label_loss,
               concept_criterion=concept_criterion, y_weight=label_loss_weight, 
               concept_weight=concept_loss_weight, device=device)

plt.plot(range(len(losses)), losses)
plt.axhline(y = 0.5, color = 'r', linestyle = 'dashed') 
plt.legend(['Training Loss', "Test Loss"]) 
plt.xlabel('Epoch') 
plt.ylabel('Loss')
plt.savefig('checkpoints/joint_cbm_'+str(sha)+"/losses.png") 
plt.show()
