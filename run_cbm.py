import torch.nn as nn
import torch
import time

from train import load_data, train
from models import *

import git

repo = git.Repo(search_parent_directories=True)
sha = repo.head.object.hexsha
print(str(sha))

device = 'cuda' if torch.cuda.is_available() else 'cpu'

data_fn = "data/college_acceptance_100001_gaussian.csv"
X_headers = ['age', 'sex', 'white', 'asian', 'black', 'hispanic', 'otherRace', 'STEM', 'percentile', 'collegeLevel']
C_headers = ['gpa', 'SAT', 'SATWriting', 'APTests', 'APScores', 'essays', 'extracurriculars', 'collegeLevel']
Y_headers = ['accepted']

train_loader, test_loader, Cy_max = load_data(X_headers, C_headers, Y_headers, data_fn, Cy_norm=True, batch_size=1024)

latents = 0

x_to_c_model = XtoCModel(len(X_headers), len(C_headers), latents, final_activation=nn.Sigmoid)
c_to_y_model = CtoYModel(len(C_headers)+latents, len(Y_headers))
model = FullModel(x_to_c_model, c_to_y_model)

label_loss = nn.BCELoss()
learning_rate = 0.001
optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)
label_loss_weight = 1
concept_loss_weight = 0.5
concept_criterion = nn.MSELoss()

losses = train(model, train_loader, len(C_headers), optimizer, y_criterion=label_loss,
               concept_criterion=concept_criterion, y_weight=label_loss_weight, 
               concept_weight=concept_loss_weight, device=device)

torch.save(model.state_dict(), 'checkpoints/joint_cbm_'+str(sha))