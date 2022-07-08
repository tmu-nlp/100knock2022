from cProfile import label
import torch
from knock71 import NeuralNetwork
from knock73 import train_dataloader, valid_dataloader, test_dataloader
from torch import nn
import matplotlib.pyplot as plt
import numpy as np

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

def plot_result(log_train, log_valid, file_name):
    fig, ax = plt.subplots(1, 2, figsize=(15, 4))
    ax[0].plot(np.array(log_train).T[0], label='train')
    ax[0].plot(np.array(log_valid).T[0], label='valid')
    ax[0].set_xlabel('epoch')
    ax[0].set_ylabel('loss')
    ax[0].set_title("Loss transition")
    ax[0].legend()
    ax[1].plot(np.array(log_train).T[1], label='train')
    ax[1].plot(np.array(log_valid).T[1], label='valid')
    ax[1].set_xlabel('epoch')
    ax[1].set_ylabel('accuracy')
    ax[1].set_title('Accuracy transition')
    ax[1].legend()
    plt.savefig(file_name)


model = NeuralNetwork(300, 4)

loss_fn = nn.CrossEntropyLoss()

oprimizer = torch.optim.SGD(model.parameters(), lr = 1e-3)

num_epochs = 100
log_train = []
log_valid = []

for epoch in range(num_epochs):
    model.train()
    for inputs, labels in train_dataloader:
        oprimizer.zero_grad()

        outputs = model(inputs)
        loss = loss_fn(outputs, labels)
        loss.backward()
        oprimizer.step()
    
    loss_train, acc_train = calculate_loss_and_accuracy(model, loss_fn, train_dataloader)
    loss_valid, acc_valid = calculate_loss_and_accuracy(model, loss_fn, valid_dataloader)
    log_train.append([loss_train, acc_train])
    log_valid.append([loss_valid, acc_valid])

    # print(f'epoch : {epoch + 1}, loss_train : {loss_train:.4f}, acc_train : {acc_train:.4f}')
    # print(f'epoch : {epoch + 1}, loss_valid : {loss_train:.4f}, acc_valid : {acc_valid:.4f}')
plot_result(log_train, log_valid, "75.png")