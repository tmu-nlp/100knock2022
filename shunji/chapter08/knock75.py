import torch
from torch import nn
import pickle
from funcs import unpickle
from matplotlib import pyplot as plt
import numpy as np
from knock73 import NewsDataset  # unpickleの時に無いとダメらしい


def calculate_loss_and_accuracy(model, criterion, loader):
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


model, dataloader_train, dataloader_valid = unpickle(
    ["SLP.pkl", "dataloader_train.pkl", "dataloader_valid.pkl"]
)

# 損失関数の定義
criterion = nn.CrossEntropyLoss()

# オプティマイザの定義
optimizer = torch.optim.SGD(model.parameters(), lr=1e-1)

# 学習
num_epochs = 100
log_train = []
log_valid = []
for epoch in range(num_epochs):
    # 訓練モードに設定
    model.train()
    for inputs, labels in dataloader_train:
        # 勾配をゼロで初期化
        optimizer.zero_grad()

        # 順伝播 + 誤差逆伝播 + 重み更新
        outputs = model(inputs)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()

    # 損失と正解率の算出
    loss_train, acc_train = calculate_loss_and_accuracy(
        model, criterion, dataloader_train
    )
    loss_valid, acc_valid = calculate_loss_and_accuracy(
        model, criterion, dataloader_valid
    )
    log_train.append([loss_train, acc_train])
    log_valid.append([loss_valid, acc_valid])

    # ログを出力
    print(
        "epoch: {}, loss_train: {}, accuracy_train: {}, loss_valid: {}, accuracy_valid: {}".format(
            epoch + 1, loss_train, acc_train, loss_valid, acc_valid
        )
    )


# 視覚化
fig, ax = plt.subplots(1, 2, figsize=(15, 5))
ax[0].plot(np.array(log_train).T[0], label="train")
ax[0].plot(np.array(log_valid).T[0], label="valid")
ax[0].set_xlabel("epoch")
ax[0].set_ylabel("loss")
ax[0].legend()
ax[1].plot(np.array(log_train).T[1], label="train")
ax[1].plot(np.array(log_valid).T[1], label="valid")
ax[1].set_xlabel("epoch")
ax[1].set_ylabel("accuracy")
ax[1].legend()
plt.show()
