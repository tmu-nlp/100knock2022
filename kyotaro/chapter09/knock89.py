from transformers import BertTokenizer, BertModel
from sklearn.model_selection import train_test_split
import torch
import time
from torch import no_grad, optim
import pandas as pd
import numpy as np
from torch import nn
from torch.utils.data import Dataset, DataLoader
from collections import defaultdict
from gensim.models import KeyedVectors
import matplotlib.pyplot as plt

class CreateDataset(Dataset):
    """入力とラベルを受け取り、toknizeしてからTensorに"""
    def __init__(self, X, y, tokenizer, max_len):
        self.X = X
        self.y = y
        self.tokenizer = tokenizer
        self.max_len = max_len
    
    def __len__(self):
        return len(self.y)
    
    def __getitem__(self, index):
        text = self.X[index]  # index番目の文を選択
        inputs = self.tokenizer.encode_plus(text, add_special_tokens=True, max_length=self.max_len, pad_to_max_length=True)  # 選択した文をIDに変換
        ids = inputs['input_ids']
        mask = inputs['attention_mask']
        item = {}
        item['ids'] = torch.LongTensor(ids)
        item['mask'] = torch.LongTensor(mask)
        item['labels'] = torch.Tensor(self.y[index])
        return item

class BERTClass(torch.nn.Module):
  def __init__(self, drop_rate, otuput_size):
    super().__init__()
    self.bert = BertModel.from_pretrained('bert-base-uncased', return_dict=False)
    self.drop = torch.nn.Dropout(drop_rate)
    self.fc = torch.nn.Linear(768, otuput_size)

  def forward(self, ids, mask):
    _, out = self.bert(ids, attention_mask=mask)
    out = self.fc(self.drop(out))
    return out


def calculate_loss_and_accuracy(model, criterion, loader, device):
    model.eval()
    loss = 0.0
    total = 0
    correct = 0
    with torch.no_grad():
        for data in loader:
            ids = data['ids'].to(device)
            mask = data['mask'].to(device)
            labels = data['labels'].to(device)
            # 前に進める
            outputs = model(ids, mask)
            # 損失関数
            loss += criterion(outputs, labels).item()
            # 正解率
            pred = torch.argmax(outputs, dim=-1).cpu().numpy()  # 予測ラベル
            labels = torch.argmax(labels, dim=-1).cpu().numpy()  # 正解ラベル
            total += len(labels)
            correct = (pred == labels).sum().item()
    return loss / len(loader), correct / total

def train_model(dataset_train, dataset_valid, batch_size, model, criterion, optimizer, num_epochs, device=None):
    model.to(device)
    # Datasetをloaderに変換
    dataloader_train = DataLoader(dataset_train, batch_size=batch_size, shuffle=True)
    dataloader_valid = DataLoader(dataset_valid, batch_size=len(dataset_valid), shuffle=False)
    # logに学習
    log_train = []
    log_valid = []
    for epoch in range(num_epochs):
        # 開始時間
        start = time.time()
        # モデルの振る舞いを変更
        model.train()
        for data in dataloader_train:
            # デバイス
            ids = data['ids'].to(device)
            mask = data['mask'].to(device)
            labels = data['labels'].to(device)
            # 勾配を初期化
            optimizer.zero_grad()
            # 前に伝搬
            outputs = model(ids, mask)
            # 損失関数
            loss = criterion(outputs, labels)
            # 逆伝搬
            loss.backward()
            optimizer.step()
        
        # lossとaccuracy
        loss_train, accuracy_train = calculate_loss_and_accuracy(model, criterion, dataloader_train, device)
        loss_valid, accuracy_valid = calculate_loss_and_accuracy(model, criterion, dataloader_valid, device)
        log_train.append([loss_train, accuracy_train])
        log_valid.append([loss_valid, accuracy_valid])
        # checkpointに保存
        torch.save({'epoch': epoch, 'model_state_dict': model.state_dict(), 'optimizer_state_dict': optimizer.state_dict()}, f'checkpoint{epoch + 1}.pt')
        # 終了時間
        end_time = time.time()
        # 出力
        print(f'epoch : {epoch + 1}, loss_train : {loss_train:.4f}, accuracy_train : {accuracy_train:.4f}, loss_valid : {loss_valid:.4f}, accuracy_valid : {accuracy_valid:.4f}, time : {(end_time - start):.4f}')
    return log_train, log_valid


if __name__ == "__main__":
    # データを読み込む
    df = pd.read_csv('newsCorpora.csv', header=None, sep='\t', names=['ID', 'TITLE', 'URL', 'PUBLISHER', 'CATEGORY', 'STORY', 'HOSTNAME', 'TIMESTAMP'])
    df = df.loc[df['PUBLISHER'].isin(['Reuters', 'Huffington Post', 'Businessweek', 'Contactmusic.com', 'Daily Mail']), ['TITLE', 'CATEGORY']]
    train, valid_test = train_test_split(df, test_size=0.2, shuffle=True, random_state=123, stratify=df['CATEGORY'])
    valid, test = train_test_split(valid_test, test_size=0.5, shuffle=True, random_state=123, stratify=valid_test['CATEGORY'])
    train.reset_index(drop=True, inplace=True)
    valid.reset_index(drop=True, inplace=True)
    test.reset_index(drop=True, inplace=True)

    # データをchapter06から持ってきて、ラベルを数値に変換
    y_train = pd.get_dummies(train, columns=['CATEGORY'])[['CATEGORY_b', 'CATEGORY_e', 'CATEGORY_t', 'CATEGORY_m']].values
    y_valid = pd.get_dummies(valid, columns=['CATEGORY'])[['CATEGORY_b', 'CATEGORY_e', 'CATEGORY_t', 'CATEGORY_m']].values
    y_test = pd.get_dummies(test, columns=['CATEGORY'])[['CATEGORY_b', 'CATEGORY_e', 'CATEGORY_t', 'CATEGORY_m']].values
    # Datasetを作成
    max_len = 20
    tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')
    dataset_train = CreateDataset(train["TITLE"], y_train, tokenizer, max_len)
    dataset_valid = CreateDataset(valid["TITLE"], y_valid, tokenizer, max_len)
    dataset_test = CreateDataset(test["TITLE"], y_test, tokenizer, max_len)
    # パラメータ
    DROP_RATE = 0.4
    OUTPUT_SIZE = 4
    BATCH_SIZE = 32
    NUM_EPOCHS = 4
    LEARNING_RATE = 2e-5
    # モデル
    model = BERTClass(DROP_RATE, OUTPUT_SIZE)
    # 損失関数
    criterion = nn.BCEWithLogitsLoss()
    # optimizer
    optimizer = torch.optim.AdamW(params=model.parameters(), lr=LEARNING_RATE)
    # 学習
    log_train, log_valid = train_model(dataset_train, dataset_valid, BATCH_SIZE, model, criterion, optimizer, NUM_EPOCHS)