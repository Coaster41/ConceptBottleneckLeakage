import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split

import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset
from tqdm import tqdm
from sklearn.metrics import f1_score, precision_score, recall_score, roc_auc_score

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
    X_train, X_test, Cy_train, Cy_test = train_test_split(X, Cy, train_size=.7, random_state=42)

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
    return train_loader, test_loader, 1


def train_one_epoch(model, train_loader, optimizer, y_criterion, concept_criterion,
                    latent_criterion, n_concepts, loss_norm=None, device='cpu', cbm_version="joint"):
    assert cbm_version == 'joint' # other version not yet implemented
    running_loss = AverageMeter()
    running_accuracy = AverageMeter()
    concept_accuracy = AverageMeter()
    
    concept_loss = AverageMeter()
    label_loss = AverageMeter()
    latent_loss = AverageMeter()

    for batch, data in enumerate(train_loader):
        X, C, y = data
        X = X.float().to(device)
        C = C.float().to(device)
        y = y.squeeze().type(torch.LongTensor).to(device)
        # Cy = Cy.to(device)
        # C = Cy[:,:n_concepts]
        # y = Cy[:, n_concepts:]

        optimizer.zero_grad()
        c_out, y_out = model(X)
        losses = []

        # print(torch.mean(c_out[:,:n_concepts], dim=0))
        # Label Loss
        if loss_norm['method'] == "weighted_exp_sum":
            losses.append(loss_norm['y_weight'] * (y_criterion(y_out, y) - loss_norm['y_utopia'])**loss_norm["exp"])
        else:
            losses.append(loss_norm['y_weight'] * y_criterion(y_out, y))
        
        # Concept Loss
        if isinstance(concept_criterion, list):
            for i in range(len(concept_criterion)):
                if isinstance(loss_norm["c_weight"], list):
                    c_weight = loss_norm["c_weight"][i]
                else:
                    c_weight = loss_norm["c_weight"]
            if loss_norm['method'] == "weighted_exp_sum":
                losses.append(c_weight * (concept_criterion[i](c_out[:,i], C[:,i]) - loss_norm['c_utopia'])**loss_norm["exp"])
            else:
                losses.append(c_weight * concept_criterion[i](c_out[:,i], C[:,i]))
        elif concept_criterion:
            if loss_norm['method'] == "weighted_exp_sum":
                losses.append(loss_norm["c_weight"] * (concept_criterion(c_out[:,:n_concepts], C) - loss_norm['c_utopia'])**loss_norm["exp"])
            else:
                losses.append(loss_norm["c_weight"] * concept_criterion(c_out[:,:n_concepts], C))
            
        # Latent Loss
        if latent_criterion:
            if loss_norm['method'] == "weighted_exp_sum":
                losses.append(loss_norm['l_weight'] * (latent_criterion(c_out[:,:n_concepts], c_out[:,n_concepts:]) - loss_norm['l_utopia'])**loss_norm["exp"])
            else:
                losses.append(loss_norm['l_weight'] * latent_criterion(c_out[:,:n_concepts], c_out[:,n_concepts:]))
            
        loss = sum(losses)
        loss.backward()
        optimizer.step()

        running_loss.update(loss.item(), X.shape[0]/train_loader.batch_size)
        running_accuracy.update(torch.mean((y == torch.argmax(y_out,1)).float()).item(), X.shape[0]/train_loader.batch_size)
        concept_accuracy.update(torch.mean((C == torch.round(c_out[:,:n_concepts])).float()).item(), X.shape[0]/train_loader.batch_size)

        label_loss.update(losses[0].item(), X.shape[0])
        concept_loss.update(losses[1].item(), X.shape[0])
        if latent_criterion:
            latent_loss.update(losses[2].item(), X.shape[0])
    
    # print(f"Label Loss: {label_loss.avg:.4f} Concept Loss: {concept_loss.avg:.4f} Latent Loss: {latent_loss.avg:.4f}")

    return running_loss.avg, running_accuracy.avg, concept_accuracy.avg, label_loss.avg, concept_loss.avg, latent_loss.avg



def train(model, train_loader, n_concepts, optimizer=None, lr=0.001, y_criterion=None, 
          concept_criterion=None, latent_criterion=None,
          loss_norm=None, epochs=50, device='cpu'): 
    if not optimizer:
        optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    if not y_criterion:
        y_criterion = nn.BCELoss()
    if not concept_criterion:
        concept_criterion = nn.MSELoss()
    
    # Train Model
    losses = []
    accuracies = []
    concept_accuracies = []
    model.train()
    for epoch in (pbar:=tqdm(range(epochs))):
        loss, accuracy, concept_accuracy, label_loss, concept_loss, latent_loss = train_one_epoch(model, train_loader, optimizer, y_criterion, 
                               concept_criterion, latent_criterion, n_concepts, loss_norm, device)
        pbar.set_description(f"Total Loss: {loss:.4f} Label Loss: {label_loss:.4f} Label Accuracy: {accuracy:.4f} Concept Loss: {concept_loss:.4f} Concept Accuracy: {concept_accuracy:.4f} Latent Loss: {latent_loss:.4f}")
        losses.append(loss)
        accuracies.append(accuracy)
        concept_accuracies.append(concept_accuracy)
    model.eval()
    return losses, accuracies, concept_accuracies


def eval(model, test_loader, n_concepts, y_criterion=None, 
          concept_criterion=None, latent_criterion=None, loss_norm=None, device='cpu'):
    if not y_criterion:
        y_criterion = nn.BCELoss()
    if not concept_criterion:
        concept_criterion = nn.MSELoss()
    if not loss_norm:
        loss_norm = {"method": "weighted_sum", "y_weight": 0.5, "c_weight": 0.5}
    
    running_loss = AverageMeter()
    running_accuracy = AverageMeter()
    concept_accuracy = AverageMeter()
    concept_loss = AverageMeter()
    label_loss = AverageMeter()
    latent_loss = AverageMeter()
    f1_score_meter = AverageMeter()
    precision = AverageMeter()
    recall = AverageMeter()
    rocauc = AverageMeter()
    model.eval()

    for batch, data in enumerate(test_loader):
        X, C, y = data
        X = X.float().to(device)
        C = C.float().to(device)
        y = y.squeeze().type(torch.LongTensor).to(device)
        # X, Cy = data
        # X = X.to(device)
        # Cy = Cy.to(device)
        # C = Cy[:,:n_concepts]
        # y = Cy[:, n_concepts:]


        c_out, y_out = model(X)
        losses = []

        # Label Loss
        if loss_norm['method'] == "weighted_exp_sum":
            losses.append(loss_norm['y_weight'] * (y_criterion(y_out, y) - loss_norm['y_utopia'])**loss_norm["exp"])
        else:
            losses.append(loss_norm['y_weight'] * y_criterion(y_out, y))
        
        # Concept Loss
        if isinstance(concept_criterion, list):
            for i in range(len(concept_criterion)):
                if isinstance(loss_norm["c_weight"], list):
                    c_weight = loss_norm["c_weight"][i]
                else:
                    c_weight = loss_norm["c_weight"]
            if loss_norm['method'] == "weighted_exp_sum":
                losses.append(c_weight * (concept_criterion[i](c_out[:,i], C[:,i]) - loss_norm['c_utopia'])**loss_norm["exp"])
            else:
                losses.append(c_weight * concept_criterion[i](c_out[:,i], C[:,i]))
        elif concept_criterion:
            if loss_norm['method'] == "weighted_exp_sum":
                losses.append(loss_norm["c_weight"] * (concept_criterion(c_out[:,:n_concepts], C) - loss_norm['c_utopia'])**loss_norm["exp"])
            else:
                losses.append(loss_norm["c_weight"] * concept_criterion(c_out[:,:n_concepts], C))
            
        # Latent Loss
        if latent_criterion:
            if loss_norm['method'] == "weighted_exp_sum":
                losses.append(loss_norm['l_weight'] * (latent_criterion(c_out[:,:n_concepts], c_out[:,n_concepts:]) - loss_norm['l_utopia'])**loss_norm["exp"])
            else:
                losses.append(loss_norm['l_weight'] * latent_criterion(c_out[:,:n_concepts], c_out[:,n_concepts:]))
            
        loss = sum(losses)

        y_np = data[1][:, n_concepts:]
        y_pred = y_out.cpu().detach().numpy()
        y_pred_round = np.round(y_pred)
        batch_size_prop = X.shape[0]/test_loader.batch_size
        running_loss.update(loss.item(), batch_size_prop)
        running_accuracy.update(torch.mean((y == torch.argmax(y_out, dim=1)).float()).item(), batch_size_prop)
        label_loss.update(losses[0].item(), batch_size_prop)
        concept_loss.update(losses[1].item(), batch_size_prop)
        if latent_criterion:
            latent_loss.update(losses[2].item(), batch_size_prop)
        concept_accuracy.update(torch.mean((C == torch.round(c_out[:,:n_concepts])).float()).item(), batch_size_prop)
        if np.array_equal(np.unique(y_np), [0,1]):
            f1_score_meter.update(f1_score(y_np, y_pred_round), batch_size_prop)
            precision.update(precision_score(y_np, y_pred_round), batch_size_prop)
            recall.update(recall_score(y_np, y_pred_round), batch_size_prop)
            rocauc.update(roc_auc_score(y_np, y_pred), batch_size_prop)


    print("Test Results:")
    print("Loss: {:.4f}, Label Accuracy: {:.4f}, Concept Accuracy: {:.4f}".format(running_loss.avg, running_accuracy.avg, concept_accuracy.avg))
    print("Label Loss: {:.4f}, Concept Loss: {:.4f} Latent Loss: {:.4f}".format(label_loss.avg, concept_loss.avg, latent_loss.avg))
    print("F1 Score: {:.4f}, Precision: {:.4f}, Recall: {:.4f}, ROCAUC: {:.4f}".format(f1_score_meter.avg, precision.avg, recall.avg, rocauc.avg))
    return running_loss.avg, running_accuracy.avg, concept_accuracy.avg, label_loss.avg, concept_loss.avg, latent_loss.avg, f1_score_meter.avg, precision.avg, recall.avg, rocauc.avg