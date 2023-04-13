from pickletools import optimize
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from torch.utils.data import DataLoader, Dataset
from torch import no_grad, optim
from collections import defaultdict
import torch
from torch import nn
import time
from knock80 import voc_id
from knock81 import RNN, CreateDataset, tokenizer

def calculate_loss_and_accuracy(model, dataset, device=None, criterion=None):
    """lossとaccuracyの計算"""
    dataloader = DataLoader(dataset, batch_size=1,
                            shuffle=False)  # iter可能な形に変換

    loss = 0.0  # lossの初期化
    total = 0  # データ総数
    correct = 0  # 正解の数

    with torch.no_grad():
        for data in dataloader:
            # デバイス
            inputs = data['inputs'].to(device)
            labels = data['labels'].to(device)

            # 前方向に伝搬
            outputs = model(inputs)

            # 損失関数
            if criterion != None:
                loss += criterion(outputs, labels).item()

            # 正解率
            pred = torch.argmax(outputs, dim=-1)  # 予測ラベル
            total += len(inputs)
            correct += (pred == labels).sum().item()  # あってたらcorrectに追加

    return loss / len(dataset), correct / total


def train_model(dataset_train, dataset_valid, batch_size, model, criterion, optimizer, num_epochs, collate_fn=None, device=None):
    """学習"""
    # デバイス
    model.to(device)

    # dataloaderを作成してiter可能に
    dataloader_train = DataLoader(
        dataset_train, batch_size=batch_size, shuffle=True, collate_fn=collate_fn)
    dataloader_valid = DataLoader(dataset_valid, batch_size=1, shuffle=False)

    # スケジューラ
    scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, num_epochs, eta_min=1e-5, last_epoch=-1)

    # 学習
    log_train = []
    log_valid = []

    for epoch in range(num_epochs):
        # 開始時間
        start = time.time()

        # modelの振る舞いを訓練用に
        model.train()

        for data in dataloader_train:
            # 勾配の初期化
            optimizer.zero_grad()

            # デバイス
            inputs = data['inputs'].to(device)
            labels = data['labels'].to(device)

            # 前方向に伝搬
            outputs = model(inputs)

            # 逆方向に伝搬
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()

        # modelの振る舞いを評価用に
        model.eval()

        # 損失と正解率の計算
        loss_train, accuracy_train = calculate_loss_and_accuracy(
            model, dataset_train, device, criterion)
        loss_valid, accuracy_valid = calculate_loss_and_accuracy(
            model, dataset_valid, device, criterion)
        log_train.append([loss_train, accuracy_train])
        log_valid.append([loss_valid, accuracy_valid])

        # checkpointに保存
        torch.save({'epoch': epoch, 'model_state_dict': model.state_dict(), 'optimizer_state_dict': optimizer.state_dict()}, f'checkpoint{epoch + 1}.pt')

        # 終了時刻
        end_time = time.time()

        # 出力
        print(f'epoch : {epoch + 1}, loss_train : {loss_train:.4f}, accuracy_train : {accuracy_train:.4f}, loss_valid : {loss_valid:.4f}, accuracy_valid : {accuracy_valid:.4f}, time : {(end_time - start):.4f}')

        # スケジューラを進める
        scheduler.step()
    
    return log_train, log_valid

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

if __name__ == "__main__":
    train = pd.read_csv("../chapter06/train.txt", sep="\t")
    valid = pd.read_csv("../chapter06/valid.txt", sep="\t")
    test = pd.read_csv("../chapter06/test.txt", sep="\t")

    frec = defaultdict(lambda: 0)  # 単語の頻度
    voc_id = defaultdict(lambda: 0)  # 単語のID

    for line in train["TITLE"]:
        words = line.strip().split()
        for word in words:
            frec[word] += 1

    frec = sorted(frec.items(), key=lambda x: x[1], reverse=True)  # 頻度順にソート

    for i, word in enumerate(frec):
        if word[1] >= 2:
            voc_id[word[0]] = i + 1


    # データをchapter06から持ってきて、ラベルを数値に変換
    category_dict = {'b': 0, 't': 1, 'e': 2, 'm': 3}

    y_train = pd.read_csv("../chapter06/train.txt", sep="\t")["CATEGORY"].map(lambda x: category_dict[x]).values
    y_valid = pd.read_csv("../chapter06/valid.txt", sep="\t")["CATEGORY"].map(lambda x: category_dict[x]).values
    y_test = pd.read_csv("../chapter06/test.txt", sep="\t")["CATEGORY"].map(lambda x: category_dict[x]).values

    # Datasetを作成
    dataset_train = CreateDataset(train["TITLE"], y_train, tokenizer, voc_id)
    dataset_valid = CreateDataset(valid["TITLE"], y_valid, tokenizer, voc_id)
    dataset_test = CreateDataset(test["TITLE"], y_test, tokenizer, voc_id)

    # パラメータ設定
    VOCAB_SIZE = len(set(voc_id.values())) + 1
    EMB_SIZE = 300
    PADDING_IDX = len(set(voc_id.values()))
    OUTPUT_SIZE = 4
    HIDDEN_SIZE = 50
    NUM_EPOCHS = 10
    LERNING_RATE = 1e-3
    BATCH_SIZE = 1

    # モデル
    model = RNN(HIDDEN_SIZE, VOCAB_SIZE, EMB_SIZE, PADDING_IDX, OUTPUT_SIZE)

    # 損失関数
    criterion = nn.CrossEntropyLoss()

    # オプティマイザ
    optimizer = torch.optim.SGD(model.parameters(), lr=LERNING_RATE)

    # 学習
    log_train, log_valid = train_model(dataset_train, dataset_valid, BATCH_SIZE, model, criterion, optimizer, NUM_EPOCHS)

    #
    # plot_result(log_train, log_valid, "82.png")
    
    # 出力
    # _, accuracy_train = calculate_loss_and_accuracy(model, dataset_train)
    # _, accuracy_valid = calculate_loss_and_accuracy(model, dataset_valid)
    # print(f'train_data accuracy : {accuracy_train:.4f}')
    # print(f'valid_data accuracy : {accuracy_valid:.4f}')