from operator import mod
import time
import torch
from torch import nn
import pickle
from torch.utils.data import Dataset
from torch.utils.data import DataLoader
from funcs import unpickle
from knock73 import NewsDataset  # unpickleの時に無いとダメらしい


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

    return {"train": log_train, "valid": log_valid}


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


if __name__ == "__main__":
    X_train = torch.load("X_train.pt")
    X_valid = torch.load("X_valid.pt")
    y_train = torch.load("y_train.pt")
    y_valid = torch.load("y_valid.pt")

    model, dataset_train, dataset_valid = unpickle(
        ["SLP.pkl", "dataset_train.pkl", "dataset_valid.pkl"]
    )
    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.SGD(model.parameters(), lr=1e-1)
    device = torch.device(
        "cuda" if torch.cuda.is_available() else "cpu"
    )  # GPUが使えるならGPU，そうでないならCPU

    for batch_size in [2**i for i in range(11)]:
        print("バッチサイズ: {}".format(batch_size))
        log = train_model(
            dataset_train,
            dataset_valid,
            batch_size,
            model,
            criterion,
            optimizer,
            1,
            device=device,
        )
