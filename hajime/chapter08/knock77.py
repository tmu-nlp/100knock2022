from knock72 import *
from torch.utils.data import Dataset
from torch.utils.data import DataLoader
from matplotlib import pyplot as plt
import numpy as np
import time


def train_model(dataset_train, dataset_valid, batch_size, model, criterion, num_epochs):
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

            outputs = model(inputs)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()
        end_time = time.time()

        loss_train, acc_train = calc_loss_acc(
            model, criterion, dataloader_train)
        loss_valid, acc_valid = calc_loss_acc(
            model, criterion, dataloader_valid)
        log_train.append([loss_train, acc_train])
        log_valid.append([loss_valid, acc_valid])

        torch.save({'epoch': epoch, 'model_state_dict': model.state_dict(
        ), 'optimizer_state_dict': optimizer.state_dict()}, f'checkpoint{epoch + 1}.pt')

        print(f'epoch: {epoch + 1}, loss_train: {loss_train:.4f}, accuracy_train: {acc_train:.4f}, loss_valid: {loss_valid:.4f}, accuracy_valid: {acc_valid:.4f}, train_time: {(end_time - start_time):.4f}sec')


model = SingleLayerPerceptronNetwork(300, 4)
criterion = nn.CrossEntropyLoss()
optimizer = torch.optim.SGD(model.parameters(), lr=1e-1)
num_epochs = 1

for batch_size in [2 ** i for i in range(12)]:
    print(f"batch_size : {batch_size}")
    train_model(dataset_train, dataset_valid,
                batch_size, model, criterion, num_epochs)

"""
batch_size : 1
epoch: 1, loss_train: 0.3263, accuracy_train: 0.8896, loss_valid: 0.3279, accuracy_valid: 0.8892, train_time: 0.4831sec
batch_size : 2
epoch: 1, loss_train: 0.3013, accuracy_train: 0.8969, loss_valid: 0.3120, accuracy_valid: 0.8952, train_time: 0.3185sec
batch_size : 4
epoch: 1, loss_train: 0.2935, accuracy_train: 0.8997, loss_valid: 0.3074, accuracy_valid: 0.8952, train_time: 0.1742sec
batch_size : 8
epoch: 1, loss_train: 0.2888, accuracy_train: 0.9012, loss_valid: 0.3019, accuracy_valid: 0.8967, train_time: 0.1149sec
batch_size : 16
epoch: 1, loss_train: 0.2872, accuracy_train: 0.9013, loss_valid: 0.3001, accuracy_valid: 0.8997, train_time: 0.0715sec
batch_size : 32
epoch: 1, loss_train: 0.2864, accuracy_train: 0.9017, loss_valid: 0.2997, accuracy_valid: 0.8997, train_time: 0.0541sec
batch_size : 64
epoch: 1, loss_train: 0.2860, accuracy_train: 0.9021, loss_valid: 0.2995, accuracy_valid: 0.8997, train_time: 0.0492sec
batch_size : 128
epoch: 1, loss_train: 0.2849, accuracy_train: 0.9021, loss_valid: 0.2994, accuracy_valid: 0.8997, train_time: 0.0352sec
batch_size : 256
epoch: 1, loss_train: 0.2857, accuracy_train: 0.9023, loss_valid: 0.2994, accuracy_valid: 0.8997, train_time: 0.0314sec
batch_size : 512
epoch: 1, loss_train: 0.2861, accuracy_train: 0.9023, loss_valid: 0.2994, accuracy_valid: 0.8997, train_time: 0.0287sec
batch_size : 1024
epoch: 1, loss_train: 0.2840, accuracy_train: 0.9024, loss_valid: 0.2993, accuracy_valid: 0.8997, train_time: 0.0293sec
batch_size : 2048
epoch: 1, loss_train: 0.2861, accuracy_train: 0.9024, loss_valid: 0.2993, accuracy_valid: 0.8997, train_time: 0.0426sec
"""
