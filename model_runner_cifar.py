import torch.nn as nn
import torch
import os
import matplotlib.pyplot as plt
import torchvision

from train import train, eval
from models import *
import json
from loss import leakage_loss, leakage_loss_simple
from cifar100_loader import CIFAR10Concepts, mobilenet_v3_large
device = 'cuda' if torch.cuda.is_available() else 'cpu'

class SyntheticDataset(torch.utils.data.Dataset):
    def __init__(self, data_fn, x_num, c_num, y_num):
        self.data = torch.load(data_fn, weights_only=True)
        self.x_num = x_num
        self.c_num = c_num
        self.y_num = y_num

    def __len__(self):
        return self.data.shape[0]
    
    def __getitem__(self, idx):
        return self.data[idx,:self.x_num].float(), self.data[idx,-self.c_num:], self.data[idx,self.x_num]
    

class BinarySigmoid(nn.Module):
    def __init__(self):
        super(BinarySigmoid, self).__init__()

    def __repr__(self):
        return 'BinarySigmoid()'

    def forward(self, x):
        x = torch.nn.functional.sigmoid(x)
        return x + torch.round(x).detach() - x.detach()

def run(expr_name, config_file):
    # Loading Experiment Config
    print(f"Training {expr_name}")
    with open(config_file) as json_file:
        config = json.load(json_file)[expr_name]['config']
    config_map = {}
    config_map["nn.Sigmoid()"] = nn.Sigmoid()
    config_map["nn.Softmax(dim=1)"] = nn.Softmax(dim=1)
    config_map["nn.CrossEntropyLoss()"] = nn.CrossEntropyLoss()
    config_map["nn.BCELoss()"] = nn.BCELoss()
    config_map["leakage_loss_simple"] = leakage_loss_simple
    config_map["BinarySigmoid()"] = BinarySigmoid()
    config_map[""] = None

    # Loading the Data
    transform = torchvision.transforms.Compose([
        torchvision.transforms.Resize((227, 227)),
        torchvision.transforms.ToTensor(),
        torchvision.transforms.Normalize(mean=[0.4914, 0.4822, 0.4465], std=[0.2023, 0.1994, 0.2010])
    ])
    cifar100_train_dataset = CIFAR10Concepts(config["data_fn"], train=True, download=False, transform=transform)
    train_loader = torch.utils.data.DataLoader(cifar100_train_dataset,
                                            batch_size=config["batch_size"],
                                            shuffle=True,
                                            num_workers=4)
    cifar100_test_dataset = CIFAR10Concepts(config["data_fn"], train=False, download=False, transform=transform)
    test_loader = torch.utils.data.DataLoader(cifar100_test_dataset,
                                            batch_size=config["batch_size"],
                                            shuffle=False,
                                            num_workers=4)

    model = mobilenet_v3_large().to(device)
    x_num = model.classifier[0].in_features

    # Load Model
    model_type = config.get('model_type')
    if model_type == 'ThreePartModel':
        x_to_c_model = XtoCModel(x_num, config["c_num"], 0, depth=config["xc_depth"], width=config["xc_width"],
                                use_relu=config["xc_use_relu"], use_sigmoid=config["xc_use_sigmoid"], 
                                final_activation=config_map.get(config["xc_final_activation"])).to(device)
        if config["l_num"]:
            x_to_l_model = XtoCModel(x_num, 0, config["l_num"], depth=config["xc_depth"], width=config["xc_width"],
                                use_relu=config["xc_use_relu"], use_sigmoid=config["xc_use_sigmoid"], 
                                final_activation=config_map.get(config["xc_final_activation"])).to(device)
        else:
            x_to_l_model = None
        c_to_y_model = CtoYModel(config["c_num"]+config["l_num"], config["y_num"], depth=config["cy_depth"], width=config["cy_width"],
                                use_relu=config["cy_use_relu"], use_sigmoid=config["cy_use_sigmoid"], 
                                final_activation=config_map.get(config["cy_final_activation"])).to(device)
        classifier = ThreePartModel(x_to_c_model, x_to_l_model, c_to_y_model, device=device).to(device)
    else:
        x_to_c_model = XtoCModel(x_num, config["c_num"], config["l_num"], depth=config["xc_depth"], width=config["xc_width"],
                                use_relu=config["xc_use_relu"], use_sigmoid=config["xc_use_sigmoid"], 
                                final_activation=config_map.get(config["xc_final_activation"])).to(device)
        c_to_y_model = CtoYModel(config["c_num"]+config["l_num"], config["y_num"], depth=config["cy_depth"], width=config["cy_width"],
                                use_relu=config["cy_use_relu"], use_sigmoid=config["cy_use_sigmoid"], 
                                final_activation=config_map.get(config["cy_final_activation"])).to(device)
        classifier = FullModel(x_to_c_model, c_to_y_model, device=device).to(device)
    
    for param in model.parameters():
        param.requires_grad = False
    model.classifier = classifier

    optimizer = torch.optim.Adam(model.parameters(), lr=config["lr"])
    # print(model)

    # Train Model
    losses, accuracies, concept_accuracy = train(model, train_loader, config["c_num"], optimizer, y_criterion=config_map.get(config["y_criterion"]),
            concept_criterion=config_map.get(config["c_criterion"]), latent_criterion=config_map.get(config["l_criterion"]), 
            loss_norm=config["loss_norm"], epochs=config["epochs"], train_method=config.get("train_method"), hard_cbm=config["hard_cbm"], device=device)
    
    # Evaluate Model
    test_loss, test_accuracy, test_concept_accuracy, test_label_loss, test_concept_loss, test_latent_loss, f1_score_meter, precision, recall, rocauc, intervention_acc, intervention_acc = eval(model, test_loader, config["c_num"], y_criterion=config_map[config["y_criterion"]],
            concept_criterion=config_map.get(config["c_criterion"]), latent_criterion=config_map.get(config["l_criterion"]), loss_norm=config["loss_norm"], hard_cbm=config["hard_cbm"], device=device)

    results = {"Loss": test_loss, "Label Accuracy": test_accuracy, "Label Loss": test_label_loss, "Concept Accuracy": test_concept_accuracy, 
            "Concept Loss": test_concept_loss, "Latent Loss": test_latent_loss, "Intervention Accuracy": intervention_acc}
    print(expr_name, results)
    torch.save(model.state_dict(), f'model_checkpoints/cifar_{expr_name}')
    with open(config_file) as json_file:
        json_dict = json.load(json_file)

    json_dict[expr_name]['results'] = results
    with open(config_file, "w") as file:
        json.dump(json_dict, file, indent=4)


def main():
    
    config_file = "experiment_results_cifar_concepts.json"
    # with open(config_file) as json_file:
    #     experiments = list(json.load(json_file).keys())
    # experiments = ["leakageLoss", "leakageDelay", "sequentialCBM"]
    # experiments = ["sequentialLatentCBM", "sequentialLeakage", "hardCBM"]
    experiments = ["hardLatentCBM", "hardLeakageCBM", "hardSequentialLatentCBM"]
    # expr_name = "sequentialLeakage"
    for expr_name in experiments:
        run(expr_name, config_file)

if __name__ == "__main__":
    main()