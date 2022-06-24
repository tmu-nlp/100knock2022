from torch.nn import functional as F
from torch import optim
import time
import torch
from torch import nn
from torch.utils.data import Dataset
from torch.utils.data import DataLoader
from funcs import unpickle
from knock76 import calculate_loss_and_accuracy
from knock73 import NewsDataset  # unpickleの時に無いとダメらしい
from matplotlib import pyplot as plt
import numpy as np


class MLP(nn.Module):
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


def calculate_loss_and_accuracy(model, criterion, loader, device):
    model.eval()
    loss = 0.0
    total = 0
    correct = 0
    with torch.no_grad():
        for inputs, labels in loader:
            inputs = inputs.to(device)
            labels = labels.to(device)
            outputs = model(inputs)
            loss += criterion(outputs, labels).item()
            pred = torch.argmax(outputs, dim=-1)
            total += len(inputs)
            correct += (pred == labels).sum().item()

    return loss / len(loader), correct / total


def train_model(
    dataset_train,
    dataset_valid,
    batch_size,
    model,
    criterion,
    optimizer,
    num_epochs,
    device=None,
):
    model.to(device)
    dataloader_train = DataLoader(dataset_train, batch_size=batch_size, shuffle=True)
    dataloader_valid = DataLoader(dataset_valid, batch_size=batch_size, shuffle=True)

    scheduler = optim.lr_scheduler.CosineAnnealingLR(
        optimizer, num_epochs, eta_min=1e-5, last_epoch=-1
    )

    log_train = []
    log_valid = []
    for epoch in range(num_epochs):
        start = time.time()

        # 訓練モードに設定
        model.train()
        for inputs, labels in dataloader_train:
            # 勾配をゼロで初期化
            optimizer.zero_grad()

            # 順伝播 + 誤差逆伝播 + 重み更新
            inputs = inputs.to(device)
            labels = labels.to(device)
            outputs = model(inputs)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()

        end = time.time()

        # 損失と正解率の算出
        loss_train, acc_train = calculate_loss_and_accuracy(
            model, criterion, dataloader_train, device=device
        )
        loss_valid, acc_valid = calculate_loss_and_accuracy(
            model, criterion, dataloader_valid, device=device
        )
        log_train.append([loss_train, acc_train])
        log_valid.append([loss_valid, acc_valid])

        torch.save(
            {
                "epoch": epoch,
                "model_state_dict": model.state_dict(),
                "optimizer_state_dict": optimizer.state_dict(),
            },
            "checkpoint{}.pt".format(epoch + 1),
        )

        # ログを出力
        print(
            "epoch: {}, loss_train: {:.4f}, accuracy_train: {:.4f}, loss_valid: {:.4f}, accuracy_valid: {:.4f}, {:.4f}sec".format(
                epoch + 1, loss_train, acc_train, loss_valid, acc_valid, end - start
            )
        )

        # 検証データの損失が3エポック連続で低下しなかった場合は学習終了
        if (
            epoch > 2
            and log_valid[epoch - 3][0]
            <= log_valid[epoch - 2][0]
            <= log_valid[epoch - 1][0]
            <= log_valid[epoch][0]
        ):
            break

        # スケジューラを1ステップ進める
        scheduler.step()

    return {"train": log_train, "valid": log_valid}


if __name__ == "__main__":
    X_train = torch.load("X_train.pt")
    X_valid = torch.load("X_valid.pt")
    y_train = torch.load("y_train.pt")
    y_valid = torch.load("y_valid.pt")

    model = MLP(300, 200, 4, 1)

    dataset_train, dataset_valid = unpickle(["dataset_train.pkl", "dataset_valid.pkl"])
    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.SGD(model.parameters(), lr=1e-3)
    device = torch.device(
        "cuda" if torch.cuda.is_available() else "cpu"
    )  # GPUが使えるならGPU，そうでないならCPU

    log = train_model(
        dataset_train, dataset_valid, 64, model, criterion, optimizer, 1000, device
    )
