from torch.nn import functional as F
from torch import nn
from torch.utils.data import DataLoader, Dataset
import torch
import matplotlib.pyplot as plt
import numpy as np
from knock73 import NewsDataset

class MLPNet(nn.Module):
  def __init__(self, input_size, mid_size, output_size, mid_layers):
    super().__init__()
    self.mid_layers = mid_layers
    self.fc = nn.Linear(input_size, mid_size)
    self.fc_mid = nn.Linear(mid_size, mid_size)
    self.fc_out = nn.Linear(mid_size, output_size) 
    self.bn = nn.BatchNorm1d(mid_size)

  def forward(self, x):
    x = F.relu(self.fc(x))
    for _ in range(self.mid_layers):
      x = F.relu(self.bn(self.fc_mid(x)))
    x = F.relu(self.fc_out(x))
    return x


def calculate_loss_and_accuracy(model, loss_fn, loader):
    model.eval()
    loss = 0.0
    total = 0
    correct = 0

    with torch.no_grad():
        for inputs, labels in loader:
            outputs = model(inputs)
            loss += loss_fn(outputs, labels).item()
            pred = torch.argmax(outputs, dim=-1)
            total += len(inputs)
            correct += (pred == labels).sum().item()
    
    return loss / len(loader), correct / total

def plot_result_multi(log, file_name):
    fig, ax = plt.subplots(1, 2, figsize=(15, 5))
    ax[0].plot(np.array(log['train']).T[0], label='train')
    ax[0].plot(np.array(log['valid']).T[0], label='valid')
    ax[0].set_xlabel('epoch')
    ax[0].set_ylabel('loss')
    ax[0].legend()
    ax[1].plot(np.array(log['train']).T[1], label='train')
    ax[1].plot(np.array(log['valid']).T[1], label='valid')
    ax[1].set_xlabel('epoch')
    ax[1].set_ylabel('accuracy')
    ax[1].legend()
    plt.savefig(file_name)

def train_model(dataset_train, dataset_valid, batch_size, model, loss_fn, optimizer, num_epochs):
    dataloader_train = DataLoader(dataset_train, batch_size=batch_size, shuffle=True)
    dataloader_valid = DataLoader(dataset_valid, batch_size=len(dataset_valid), shuffle=True)

    log_train = []
    log_valid = []

    for epoch in range(num_epochs):
        model.train()

        for inputs, labels in dataloader_train:
            optimizer.zero_grad()

            outputs = model.forward(inputs)
            loss = loss_fn(outputs, labels)
            loss.backward()
            optimizer.step()
        
        loss_train, acc_train = calculate_loss_and_accuracy(model, loss_fn, dataloader_train)
        loss_valid, acc_valid = calculate_loss_and_accuracy(model, loss_fn, dataloader_valid)
        log_train.append([loss_train, acc_train])
        log_valid.append([loss_valid, acc_valid])

        # print(f'epoch: {epoch + 1}, loss_train: {loss_train:.4f}, accuracy_train: {acc_train:.4f}, loss_valid: {loss_valid:.4f}, accuracy_valid: {acc_valid:.4f}')

    return {'train' : log_train, 'valid' : log_valid}


X_train = torch.load('./tensor/X_train.pt')
Y_train = torch.load('./tensor/Y_train.pt')
X_valid = torch.load('./tensor/X_valid.pt')
Y_valid = torch.load('./tensor/Y_valid.pt')
X_test = torch.load('./tensor/X_test.pt')
Y_test = torch.load('./tensor/Y_test.pt')

dataset_train = NewsDataset(X_train, Y_train)
dataset_valid = NewsDataset(X_valid, Y_valid)

model = MLPNet(300, 200, 4, 1)

loss_fn = nn.CrossEntropyLoss()
optimizer = torch.optim.SGD(model.parameters(), lr=1e-3)

log = train_model(dataset_train, dataset_valid, 64, model, loss_fn, optimizer, 100)

plot_result_multi(log, "79.png")
