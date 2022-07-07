from torch import nn
import torch
from knock71 import *
from torch.utils.data import Dataset
from torch.utils.data import DataLoader


class NewsDataset(Dataset):
    def __init__(self, x, y):
        self.x = x
        self.y = y

    def __len__(self):
        return len(self.y)

    def __getitem__(self, idx):
        return [self.x[idx], self.y[idx]]


def calc_acc(model, loader):
    model.eval()
    total = 0
    correct = 0
    with torch.no_grad():
        for inputs, labels in loader:
            outputs = model(inputs)
            pred = torch.argmax(outputs, dim=-1)
            total += len(inputs)
            correct += (pred == labels).sum().item()

        return correct / total


def calc_loss_acc(model, criterion, loader):
    model.eval()
    loss = 0.0
    total = 0
    correct = 0
    with torch.no_grad():
        for inputs, labels in loader:
            outputs = model(inputs)
            loss += criterion(outputs, labels).item()
            pred = torch.argmax(outputs, dim=-1)
            total += len(inputs)
            correct += (pred == labels).sum().item()

        return loss / len(loader), correct / total


X_train = torch.load("X_train.pt")
X_valid = torch.load("X_valid.pt")
X_test = torch.load("X_test.pt")
y_train = torch.load("y_train.pt")
y_valid = torch.load("y_valid.pt")
y_test = torch.load("y_test.pt")

dataset_train = NewsDataset(X_train, y_train)
dataset_valid = NewsDataset(X_valid, y_valid)
dataset_test = NewsDataset(X_test, y_test)

dataloader_train = DataLoader(dataset_train, batch_size=1, shuffle=True)
dataloader_valid = DataLoader(
    dataset_valid, batch_size=len(dataset_valid), shuffle=False)
dataloader_test = DataLoader(
    dataset_test, batch_size=len(dataset_test), shuffle=False)

criterion = nn.CrossEntropyLoss()
l_1 = criterion(model(X_train[:1]), y_train[:1])
model.zero_grad()
l_1.backward()
# print(f'loss: {l_1:.4f}')
# print(f'grad:\n{model.fc.weight.grad}')

l = criterion(model(X_train[:4]), y_train[:4])
model.zero_grad()
l.backward()
# print(f'loss: {l:.4f}')
# print(f'grad:\n{model.fc.weight.grad}')

"""
loss: 1.4949
grad:
tensor([[ 0.0009,  0.0026, -0.0028,  ..., -0.0005,  0.0041, -0.0021],
        [ 0.0149,  0.0440, -0.0471,  ..., -0.0086,  0.0697, -0.0356],
        [-0.0161, -0.0476,  0.0509,  ...,  0.0093, -0.0752,  0.0384],
        [ 0.0003,  0.0009, -0.0010,  ..., -0.0002,  0.0015, -0.0007]])
loss: 1.1154
grad:
tensor([[ 0.0011,  0.0003, -0.0017,  ..., -0.0021,  0.0013, -0.0012],
        [-0.0083,  0.0300, -0.0231,  ..., -0.0401,  0.0255,  0.0120],
        [ 0.0059, -0.0297,  0.0255,  ...,  0.0439, -0.0272, -0.0094],
        [ 0.0012, -0.0006, -0.0007,  ..., -0.0016,  0.0004, -0.0014]])
"""
