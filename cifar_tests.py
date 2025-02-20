import torch.nn as nn
import torch
import os
import matplotlib.pyplot as plt

from train import train, eval
from models import *
from loss import leakage_loss, leakage_loss_simple

class Cifar10Dataset(torch.utils.data.Dataset):
    def __init__(self, root, train=True, n_concepts=32):
        self.root = root
        self.base_folder = 'cifar-10-batches-py'
        
        file_name = ('train_features', 'test_features')
        file_path = os.path.join(self.root, self.base_folder, file_name[int(train)])
        self.features = torch.load(file_path, weights_only=True)
        
        file_name = ('train_labels', 'test_labels')
        file_path = os.path.join(self.root, self.base_folder, file_name[int(train)])
        self.labels = torch.load(file_path, weights_only=True)

        file_name = ('train_concepts', 'test_concepts')
        file_path = os.path.join(self.root, self.base_folder, file_name[int(train)])
        if not os.path.isfile(file_path):
            self.concepts = torch.zeros(self.labels.shape[0], n_concepts)
        else:
            self.concepts = torch.load(file_path, weights_only=True)

    def __getitem__(self, index):
        return self.features[index], self.concepts[index], self.labels[index]
    
    def __len__(self):
        return self.features.shape[0]
    
    def update_concepts(self, index, concepts):
        self.concepts[index] = concepts
    
    def save_concepts(self):
        file_name = ('train_concepts', 'test_concepts')
        file_path = os.path.join(self.root, self.base_folder, file_name[int(self.train)])
        torch.save(self.concepts, file_path)



def main():
    print("load data start")
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    x_num = 2048
    c_num = 5
    y_num = 10
    cifar10_train_dataset = Cifar10Dataset('../data/', train=True, n_concepts=c_num)
    train_loader = torch.utils.data.DataLoader(cifar10_train_dataset,
                                            batch_size=1024,
                                            shuffle=True,
                                            num_workers=0)
    cifar10_train_dataset = Cifar10Dataset('../data/', train=False, n_concepts=c_num)
    test_loader = torch.utils.data.DataLoader(cifar10_train_dataset,
                                            batch_size=1024,
                                            shuffle=False,
                                            num_workers=0)
    x_to_c_model = XtoCModel(x_num, c_num, latents=0, width=64, depth=2, final_activation=nn.Sigmoid())
    c_to_y_model = CtoYModel(c_num, y_num, width=128, depth=1, final_activation=nn.Softmax(dim=1)).to(device)
    model = FullModel(x_to_c_model, c_to_y_model, device=device).to(device)
    print(model)
    # model.load_state_dict(torch.load('model_checkpoints/cifar10_feature_weights.pt'))
    losses, accuracies, concept_accuracy = train(model, train_loader, c_num, torch.optim.Adam(model.parameters(), lr=0.001), y_criterion=nn.CrossEntropyLoss(),
            concept_criterion=None, latent_criterion=None, 
            loss_norm={'method':"weighted_sum", "y_weight": 1, "c_weight": 0, "l_weight": 0}, epochs=20, train_method=None, hard_cbm=True, device=device)
    torch.save(model.state_dict(), "model_checkpoints/cifar10_feature_weights.pt")
    test_loss, test_accuracy, test_concept_accuracy, test_label_loss, test_concept_loss, test_latent_loss, f1_score_meter, precision, recall, rocauc, intervention_acc = eval(model, test_loader, c_num, y_criterion=nn.CrossEntropyLoss(),
            concept_criterion=None, latent_criterion=None, loss_norm={'method':"weighted_sum", "y_weight": 1, "c_weight": 0, "l_weight": 0}, hard_cbm=True, device=device)
        

if __name__ == "__main__":
    main()