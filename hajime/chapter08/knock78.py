# https://colab.research.google.com/drive/1CYOx1AfE_rhjIs6WmwhISwkxPHgsIDkm?usp=sharing
# google colab上でやったものを流用

from torch import nn
import torch
from torch.utils.data import Dataset
from torch.utils.data import DataLoader
from matplotlib import pyplot as plt
import numpy as np
import time


class SingleLayerPerceptronNetwork(nn.Module):
    def __init__(self, input_size, output_size):
        super().__init__()
        self.fc = nn.Linear(input_size, output_size, bias=False)
        nn.init.normal_(self.fc.weight, 0.0, 1.0)

    def forward(self, x):
        x = self.fc(x)
        return x


X_train = torch.load("X_train.pt")
model = SingleLayerPerceptronNetwork(300, 4)
y_hat_1 = torch.softmax(model(X_train[:1]), dim=-1)
Y_hat = torch.softmax(model.forward(X_train[:4]), dim=-1)


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


def calc_loss_acc(model, criterion, loader, device):  # deviceを引数に追加
    model.eval()
    loss = 0.0
    total = 0
    correct = 0
    with torch.no_grad():
        for inputs, labels in loader:
            inputs = inputs.to(device)  # deviceに移動
            labels = labels.to(device)  # deviceに移動
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

l = criterion(model(X_train[:4]), y_train[:4])
model.zero_grad()
l.backward()


def train_model(dataset_train, dataset_valid, batch_size, model, criterion, num_epochs, device=None):
    model.to(device)  # 追加

    dataloader_train = DataLoader(
        dataset_train, batch_size=batch_size, shuffle=True)
    dataloader_valid = DataLoader(
        dataset_valid, batch_size=len(dataset_valid), shuffle=False)

    log_train = []
    log_valid = []

    for epoch in range(num_epochs):
        start_time = time.time()

        model.train()
        loss_train = 0.0
        for inputs, labels in dataloader_train:
            optimizer.zero_grad()

            inputs = inputs.to(device)  # deviceに移動
            labels = labels.to(device)  # deviceに移動
            outputs = model.forward(inputs)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()
        end_time = time.time()

        loss_train, acc_train = calc_loss_acc(
            model, criterion, dataloader_train, device)
        loss_valid, acc_valid = calc_loss_acc(
            model, criterion, dataloader_valid, device)
        log_train.append([loss_train, acc_train])
        log_valid.append([loss_valid, acc_valid])

        torch.save({'epoch': epoch, 'model_state_dict': model.state_dict(
        ), 'optimizer_state_dict': optimizer.state_dict()}, f'checkpoint{epoch + 1}.pt')

        print(f'epoch: {epoch + 1}, loss_train: {loss_train:.4f}, accuracy_train: {acc_train:.4f}, loss_valid: {loss_valid:.4f}, accuracy_valid: {acc_valid:.4f}, train_time: {(end_time - start_time):.4f}sec')


model = SingleLayerPerceptronNetwork(300, 4)
criterion = nn.CrossEntropyLoss()
optimizer = torch.optim.SGD(model.parameters(), lr=1e-1)
num_epochs = 1
device = torch.device("cuda")  # deviceを指定

for batch_size in [2 ** i for i in range(12)]:
    print(f"batch_size : {batch_size}")
    train_model(dataset_train, dataset_valid,
                batch_size, model, criterion, num_epochs, device)
"""
batch_size : 1
epoch: 1, loss_train: 0.3296, accuracy_train: 0.8864, loss_valid: 0.3435, accuracy_valid: 0.8832, train_time: 6.0722sec
batch_size : 2
epoch: 1, loss_train: 0.3076, accuracy_train: 0.8954, loss_valid: 0.3254, accuracy_valid: 0.8952, train_time: 3.0445sec
batch_size : 4
epoch: 1, loss_train: 0.2954, accuracy_train: 0.8992, loss_valid: 0.3159, accuracy_valid: 0.8930, train_time: 1.5249sec
batch_size : 8
epoch: 1, loss_train: 0.2919, accuracy_train: 0.9001, loss_valid: 0.3135, accuracy_valid: 0.8952, train_time: 0.8329sec
batch_size : 16
epoch: 1, loss_train: 0.2900, accuracy_train: 0.9008, loss_valid: 0.3115, accuracy_valid: 0.8975, train_time: 0.4207sec
batch_size : 32
epoch: 1, loss_train: 0.2891, accuracy_train: 0.9011, loss_valid: 0.3112, accuracy_valid: 0.8960, train_time: 0.2404sec
batch_size : 64
epoch: 1, loss_train: 0.2887, accuracy_train: 0.9011, loss_valid: 0.3111, accuracy_valid: 0.8960, train_time: 0.1392sec
batch_size : 128
epoch: 1, loss_train: 0.2885, accuracy_train: 0.9012, loss_valid: 0.3110, accuracy_valid: 0.8960, train_time: 0.0913sec
batch_size : 256
epoch: 1, loss_train: 0.2882, accuracy_train: 0.9012, loss_valid: 0.3110, accuracy_valid: 0.8960, train_time: 0.0732sec
batch_size : 512
epoch: 1, loss_train: 0.2884, accuracy_train: 0.9013, loss_valid: 0.3109, accuracy_valid: 0.8960, train_time: 0.0600sec
batch_size : 1024
epoch: 1, loss_train: 0.2888, accuracy_train: 0.9013, loss_valid: 0.3109, accuracy_valid: 0.8960, train_time: 0.0538sec
batch_size : 2048
epoch: 1, loss_train: 0.2881, accuracy_train: 0.9013, loss_valid: 0.3109, accuracy_valid: 0.8960, train_time: 0.0994sec
"""
