import torch
from knock71 import NeuralNetwork
from knock73 import train_dataloader, valid_dataloader, test_dataloader
from torch import nn
import matplotlib.pyplot as plt
import numpy as np
import time


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

def train_batch(num_epochs, model, optimizer, loss_fn, b):
    for epoch in range(num_epochs):
        model.train()
        for i, (inputs, labels) in enumerate(train_dataloader):
            if i % b == 0:
                optimizer.zero_grad()

                outputs = model(inputs)
                loss = loss_fn(outputs, labels)
                loss.backward()
                optimizer.step()

        loss_train, acc_train = calculate_loss_and_accuracy(
            model, loss_fn, train_dataloader)
        loss_valid, acc_valid = calculate_loss_and_accuracy(
            model, loss_fn, valid_dataloader)
        log_train.append([loss_train, acc_train])
        log_valid.append([loss_valid, acc_valid])

        torch.save({'epoch': epoch, 'model': model.state_dict(
        ), 'optimizer': optimizer.state_dict()}, f'./checkpoints/checkpoint{epoch}.pt')

B = [1, 2, 4, 8, 10]

for b in B:
    model = NeuralNetwork(300, 4)
    loss_fn = nn.CrossEntropyLoss()
    optimizer = torch.optim.SGD(model.parameters(), lr=1e-1)
    num_epochs = 10
    log_train = []
    log_valid = []

    start = time.time()

    train_batch(num_epochs, model, optimizer, loss_fn, b)

    elapsed_time = time.time() - start
    
    print(f'{b} elapsed_time : {elapsed_time} [sec]')


"""
1 elapsed_time : 8.484923839569092 [sec]
2 elapsed_time : 6.639693975448608 [sec]
4 elapsed_time : 5.35027003288269 [sec]
8 elapsed_time : 4.860385894775391 [sec]
10 elapsed_time : 4.589254140853882 [sec]
"""