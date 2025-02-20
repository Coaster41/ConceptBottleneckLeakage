import torch
import torchvision
import os
import torch.nn as nn
# import torchvision.models.mobilenetv3
from models import FullModel, ThreePartModel, Cifar10CnnModel, CtoYModel
import time
import torchvision.transforms as transforms
import PIL


class CIFAR100Concepts(torchvision.datasets.CIFAR100):
    def __init__(self, root, train=True, transform=None, target_transform=None, download=False, n_concepts=32):
        super().__init__(root, train=train, transform=transform, target_transform=target_transform, download=download)
        file_name = ('train_concepts', 'test_concepts')
        file_path = os.path.join(self.root, self.base_folder, file_name[int(train)])
        if not os.path.isfile(file_path):
            self.concepts = torch.zeros(len(self.targets), n_concepts)
        else:
            self.concepts = torch.load(file_path, weights_only=True)
        self.get_index = False

    def __getitem__(self, index):
        img, target = super().__getitem__(index)
        concept = self.concepts[index]
        if self.get_index:
            return img, concept, target, index
        return img, concept, target
    
    def update_concepts(self, index, concepts):
        self.concepts[index] = concepts
    
    def save_concepts(self):
        file_name = ('train_concepts', 'test_concepts')
        file_path = os.path.join(self.root, self.base_folder, file_name[int(self.train)])
        torch.save(self.concepts, file_path)

class CIFAR10Concepts(torchvision.datasets.CIFAR10):
    def __init__(self, root, train=True, transform=None, target_transform=None, download=False, n_concepts=32):
        super().__init__(root, train=train, transform=transform, target_transform=target_transform, download=download)
        file_name = ('train_concepts', 'test_concepts')
        file_path = os.path.join(self.root, self.base_folder, file_name[int(train)])
        if not os.path.isfile(file_path):
            self.concepts = torch.zeros(len(self.targets), n_concepts)
        else:
            self.concepts = torch.load(file_path, weights_only=True)
        self.get_index = False

    def __getitem__(self, index):
        img, target = super().__getitem__(index)
        concept = self.concepts[index]
        if self.get_index:
            return img, concept, target, index
        return img, concept, target
    
    def update_concepts(self, index, concepts):
        self.concepts[index] = concepts
    
    def save_concepts(self):
        file_name = ('train_concepts', 'test_concepts')
        file_path = os.path.join(self.root, self.base_folder, file_name[int(self.train)])
        torch.save(self.concepts, file_path)


# Helpers
class AverageMeter(object):
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

def multi_label_acc(y_hat, y):
    return torch.count_nonzero(torch.round(y_hat) == y).item() / y.numel()

def classification_acc(y_hat, y):
    return torch.count_nonzero(torch.argmax(y_hat, dim=1) == y).item() / y.numel()

class BinarySigmoid(nn.Module):
    def __init__(self):
        super(BinarySigmoid, self).__init__()

    def __repr__(self):
        return 'BinarySigmoid()'

    def forward(self, x):
        x = torch.nn.functional.sigmoid(x)
        return x + torch.round(x).detach() - x.detach()

# class Default_Model(torch.nn.Module):
#     def __init__(self, num_x, num_y, depth=3, width=16, activation_func=nn.ReLU()):
#         super(Default_Model, self).__init__()
#         self.activation = activation_func
#         self.linears = []
#         prev_width = num_x
#         for i in range(depth):
#             if i == depth - 1:
#                 self.linears.append(nn.Linear(prev_width, num_y))
#             else:
#                 if isinstance(width, list):
#                     self.linears.append(nn.Linear(prev_width, width[i]))
#                     prev_width = width[i]
#                 else:
#                     self.linears.append(nn.Linear(prev_width, width))
#                     prev_width = width
#         self.linears = nn.Sequential(*self.linears)
    
#     def forward(self, x):
#         for layer_num, layer in enumerate(self.linears):
#             x = layer(x)
#             if layer_num < len(self.linears)-1:
#                 x = self.activation(x)
#         return x


# class FullModel2(torch.nn.Module):
#     def __init__(self, x_to_c_model, c_to_y_model, concept_activation=nn.Sigmoid(), label_activation=nn.Sigmoid()):
#         super(FullModel2, self).__init__()
#         self.x_to_c_model = x_to_c_model
#         self.c_to_y_model = c_to_y_model
#         self.concept_activation = concept_activation
#         self.label_activation = label_activation
    
#     def forward(self, x):
#         c_out = self.x_to_c_model(x)
#         if self.concept_activation:
#             c_out = self.concept_activation(c_out)
#         y_out = self.c_to_y_model(c_out)
#         if self.label_activation:
#             y_out = self.label_activation(y_out)
#         return c_out, y_out

# class MobileNetConcepts(torchvision.models.mobilenetv3.MobileNetV3):    
#     def forward(self, x, use_latents=True, hard_cbm=False):
#         x = self.features(x)

#         x = self.avgpool(x)
#         x = torch.flatten(x, 1)

#         if isinstance(self.classifier, FullModel) or isinstance(self.classifier, ThreePartModel):
#             c, x = self.classifier(x, use_latents=use_latents, hard_cbm=hard_cbm)
#         else:
#             c, x = self.classifier(x)

#         return c, x
    
#     def forward_concepts(self, c, l, hard_cbm=False):
#         return self.classifier.forward_concepts(c, l, hard_cbm=hard_cbm)
    
#     def freeze_x_to_c(self, freeze=True):
#         return self.classifier.freeze_x_to_c(freeze=freeze)


def train_concept_model(model, train_loader, device='cpu'):
    y_criterion = nn.CrossEntropyLoss()
    # optimizer = torch.optim.Adam(model.parameters(), lr=0.01)
    optimizer = torch.optim.SGD(model.parameters(), lr=0.01, momentum=0.9, weight_decay=5e-4)
    loss_meter = AverageMeter()
    acc_meter = AverageMeter()
    epochs = 50
    model.train()
    for epoch in range(epochs):
        loss_meter.reset()
        acc_meter.reset()
        for batch, (x, _, y) in enumerate(train_loader):
            x = x.float().to(device)
            y = y.squeeze().type(torch.LongTensor).to(device)
            
            c_pred, y_pred = model(x)
            # y_pred = model(x)

            y_loss = y_criterion(y_pred, y)
            loss = y_loss
            
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            loss_meter.update(loss.item(), x.shape[0])
            acc_meter.update(classification_acc(y_pred, y), x.shape[0])
        
        print(f"Epoch: {epoch} Loss: {loss_meter.avg} y_acc: {acc_meter.avg}")


def test_concept_model(model, dataset, device='cpu', update_concepts=False):
    model.eval()
    dataset.get_index=update_concepts
    data_loader = torch.utils.data.DataLoader(dataset,
                                            batch_size=64,
                                            shuffle=False,
                                            num_workers=4)
    y_criterion = nn.CrossEntropyLoss()
    loss_meter = AverageMeter()
    acc_meter = AverageMeter()
    with torch.no_grad():
        for batch, data in enumerate(data_loader):
            if update_concepts:
                x, c, y, index = data
            else:
                x, c, y = data
            x = x.float().to(device)
            y = y.squeeze().type(torch.LongTensor).to(device)
            
            c_pred, y_pred = model(x)
            # y_pred = model(x)

            y_loss = y_criterion(y_pred, y)
            loss = y_loss

            loss_meter.update(loss.item(), x.shape[0])
            acc_meter.update(classification_acc(y_pred, y), x.shape[0])

            if update_concepts:
                dataset.update_concepts(index, c_pred.detach().cpu())
    
    print(f"Test Loss: {loss_meter.avg} y_acc: {acc_meter.avg}")
    if update_concepts:
        dataset.save_concepts()
        dataset.get_index=False

# def _mobilenet_v3(
#     inverted_residual_setting,
#     last_channel,
#     weights,
#     progress,
# ):

#     model = MobileNetConcepts(inverted_residual_setting, last_channel)

#     if weights is not None:
#         model.load_state_dict(weights.get_state_dict(progress=progress, check_hash=True))

#     return model

# def mobilenet_v3_large():
#     weights = torchvision.models.mobilenetv3.MobileNet_V3_Large_Weights.verify(torchvision.models.mobilenetv3.MobileNet_V3_Large_Weights.IMAGENET1K_V2)

#     inverted_residual_setting, last_channel = torchvision.models.mobilenetv3._mobilenet_v3_conf("mobilenet_v3_large")
#     return _mobilenet_v3(inverted_residual_setting, last_channel, weights, True)


def main():
    print("load data start")
    start_time = time.time()
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    train_transform = transforms.Compose([
        transforms.RandomCrop(32, padding=4),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
        transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)),
        ])
    test_transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)),
        ])
    # train_transform = transforms.Compose(
    #     [transforms.RandomHorizontalFlip(p=0.5),
    #     transforms.RandomAffine(degrees=(-5, 5), translate=(0.1, 0.1), scale=(0.9, 1.1)),
    #     transforms.ToTensor(),
    #     transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))])
    # test_transform = transforms.Compose(
    #     [transforms.ToTensor(),
    #     transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))])
    c_num = 2048
    y_num = 10
    cifar10_train_dataset = CIFAR10Concepts('../data/', train=True, download=False, transform=train_transform, n_concepts=c_num)
    train_loader = torch.utils.data.DataLoader(cifar10_train_dataset,
                                            batch_size=64,
                                            shuffle=True,
                                            num_workers=4)
    cifar10_test_dataset = CIFAR10Concepts('../data/', train=False, download=False, transform=test_transform, n_concepts=c_num)
    print("finished loading data", time.time() - start_time)
    # x_to_c_model = Cifar10CnnModel(c_num, 0, final_activation=nn.ReLU()).to(device)
    x_to_c_model = torchvision.models.resnet50(pretrained=True)
    x_to_c_model.conv1 = nn.Conv2d(3, 64, kernel_size=3, stride=1, padding=1, bias=False)
    num_features = x_to_c_model.fc.in_features
    c_num = num_features
    print('num_features', num_features)
    x_to_c_model.fc = nn.ReLU()
    model = x_to_c_model.to(device)
    c_to_y_model = CtoYModel(c_num, y_num, width=128, depth=0, final_activation=nn.Softmax(dim=1)).to(device)
    model = FullModel(x_to_c_model, c_to_y_model, device=device, concepts=c_num, latents=0).to(device)
    # print(model)
    print("start training data", time.time() - start_time)
    train_concept_model(model, train_loader, device)
    # model.load_state_dict(torch.load("cifar10_weights.pt", weights_only=True))
    test_concept_model(model, cifar10_test_dataset, device)
    # save = input("Save model concepts? (y or n) ")
    # if save == 'y':
    torch.save(model.state_dict(), "cifar10_weights.pt")
    test_concept_model(model, cifar10_train_dataset, device, update_concepts=True)
    test_concept_model(model, cifar10_test_dataset, device, update_concepts=True)
        

if __name__ == "__main__":
    print("start start")
    main()