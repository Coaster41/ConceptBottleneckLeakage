import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split

import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset
import tqdm

class AverageMeter(object):
    """
    Computes and stores the average and current value
    """

    def __init__(self):
        self.reset()

    def reset(self):
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0

    def update(self, val, n=1):
        self.val = val
        self.sum += val * n
        self.count += n
        self.avg = self.sum / self.count

def load_data(X_headers, C_headers, Y_headers, file_name, Cy_norm=True, batch_size=1024):
    df = pd.read_csv(file_name, index_col=0)
    df.drop(df.index[-1],inplace=True)

    CY_headers = C_headers + Y_headers

    X = df.loc[:,X_headers].values
    Cy = df.loc[:,CY_headers].values
    if Cy_norm:
        Cy_max = np.max(Cy, axis=0)
        Cy = Cy / Cy_max
    X_train, X_test, Cy_train, Cy_test = train_test_split(X, Cy, train_size=.7)

    train_X = torch.Tensor(X_train)
    train_Cy = torch.Tensor(Cy_train)
    test_X = torch.Tensor(X_test)
    test_Cy = torch.Tensor(Cy_test)
    training_dataset = TensorDataset(train_X, train_Cy)
    train_loader = DataLoader(training_dataset, batch_size=batch_size, shuffle=True)
    testing_dataset = TensorDataset(test_X, test_Cy)
    test_loader = DataLoader(testing_dataset, batch_size=batch_size, shuffle=True)

    if Cy_norm:
        return train_loader, test_loader, Cy_max
    return train_loader, test_loader


def train_one_epoch(model, train_loader, optimizer, y_criterion, concept_criterion,
                    latent_criterion, y_weight, concept_weight, latent_weight, 
                    n_concepts, device='cpu', cbm_version="joint"):
    assert cbm_version == 'joint' # other version not yet implemented
    running_loss = AverageMeter()

    for batch, data in enumerate(train_loader):
        X, Cy = data
        X = X.to(device)
        Cy = Cy.to(device)
        C = Cy[:,:n_concepts]
        y = Cy[:, n_concepts:]

        optimizer.zero_grad()
        c_out, y_out = model(X)
        losses = []

        # Label Loss
        losses.append(y_weight * y_criterion(y_out, y))
        
        # Concept Loss
        if isinstance(concept_criterion, list):
            for i in range(len(concept_criterion)):
                if isinstance(concept_weight, list):
                    c_weight = concept_weight[i]
                else:
                    c_weight = concept_weight
                losses.append(c_weight * concept_criterion[i](c_out[:,i], C[:,i]))
        else:
            losses.append(concept_weight * concept_criterion(c_out[:,:n_concepts], C))
            
        # Latent Loss
        if latent_criterion:
            losses.append(latent_weight * latent_criterion(c_out[:,:n_concepts], c_out[:,n_concepts:]))
            
        loss = sum(losses)
        loss.backward()
        optimizer.step()

        running_loss.update(loss.item(), X.shape[0]/train_loader.batch_size)

    return running_loss.avg



def train(model, train_loader, n_concepts, optimizer=None, lr=0.001, y_criterion=None, 
          concept_criterion=None, latent_criterion=None, y_weight=1, 
          concept_weight=0.1, latent_weight=0.5, epochs=50, device='cpu'): 
    if not optimizer:
        optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    if not y_criterion:
        y_criterion = nn.BCELoss()
    if not concept_criterion:
        concept_criterion = nn.MSELoss()
    
    # Train Model
    losses = []
    model.train()
    for epoch in tqdm(range(epochs)):
        loss = train_one_epoch(model, train_loader, optimizer, y_criterion, 
                               concept_criterion, latent_criterion, y_weight, 
                               concept_weight, latent_weight, n_concepts, device)
        losses.append(loss)
    model.eval()
    return losses
    
