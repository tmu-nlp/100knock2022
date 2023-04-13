#denebの100knockっていう仮想環境で動かす
import torch 
import numpy as np
from torch import nn
from torch.utils.data import Dataset
from torch.utils.data import DataLoader
import pandas as pd
import time
from tqdm import tqdm
from matplotlib import pyplot as plt
from transformers import BertTokenizer, BertModel

class BERTClass(torch.nn.Module):
    def __init__(self, drop_rate, output_size):
        super().__init__()
        self.bert = BertModel.from_pretrained('bert-base-uncased', return_dict=False) #return_dict=False を書かなきゃv4では動かない
        self.drop = torch.nn.Dropout(drop_rate) #BERTの出力ベクトルを受け取る
        self.fc = torch.nn.Linear(768, output_size) #全結合層

    def forward(self, ids, mask):
        _, out = self.bert(ids, attention_mask=mask)
        out = self.fc(self.drop(out))
        return out
    
class MakeDataset(Dataset):
    def __init__(self, X, y, tokenizer, max_len):
        self.X = X
        self.y = y
        self.tokenizer = tokenizer
        self.max_len = max_len

    def __len__(self):  # len()でサイズを返す
        return len(self.y)

    def __getitem__(self, index):
        """tokenizerで入力テキストの前処理を行い、指定した最長系列長までパディングしてから単語IDに変換する"""
        text = self.X[index]
        inputs = self.tokenizer.encode_plus(
            text, 
            add_special_tokens=True, 
            max_length=self.max_len, 
            pad_to_max_length=True
        )

        ids = inputs['input_ids']
        mask = inputs['attention_mask']
        # mask : パディングした位置は0、それ以外は1
        return {
            'ids': torch.LongTensor(ids),
            'mask': torch.LongTensor(mask),
            'labels': torch.Tensor(self.y[index])
        }

def calculate_loss_acc(model, dataset, device=None, criterion=None):
    model.eval()
    dataloader = DataLoader(dataset, batch_size=1, shuffle=False)
    loss = 0
    total = 0
    corr = 0
    with torch.no_grad():
        for data in dataloader:
            ids = data['ids'].to(device)
            mask = data['mask'].to(device)
            labels = data['labels'].to(device)

            outputs = model(ids, mask) # 順伝播
            if criterion != None: # 損失計算
                loss += criterion(outputs, labels).item()
            # 正解率を計算する
            pred = torch.argmax(outputs, dim=-1).cpu().numpy() #バッチサイズ分の予測ラベル
            labels = torch.argmax(labels, dim=-1).cpu().numpy() #バッチサイズ分の正解ラベル
            total += len(labels)
            corr += (pred == labels).sum().item()
    return loss/len(dataset), corr/total

def train_model(dataset_train, dataset_valid, model, criterion, optimizer, batch_size=1, epochs=10, device=None):
    model.to(device)

    # dataloaderを作成する
    dataloader_train = DataLoader(dataset_train, batch_size=batch_size, shuffle=True)
    dataloader_valid = DataLoader(dataset_valid, batch_size=1, shuffle=False)

    log_train = []
    log_valid = []
    for i in tqdm(range(epochs)):
        s_time = time.time()

        model.train() #訓練モード
        for data in dataloader_train:
            optimizer.zero_grad()
            ids = data['ids'].to(device)
            mask = data['mask'].to(device)
            labels = data['labels'].to(device)

            outputs = model(ids, mask) #順伝藩
            loss = criterion(outputs, labels) 
            loss.backward() #逆伝藩
            optimizer.step() #重み更新

        model.eval() #評価モード

        #損失と正解率の算出
        loss_train, acc_train = calculate_loss_acc(model, dataset_train, device, criterion=criterion)
        loss_valid, acc_valid = calculate_loss_acc(model, dataset_valid, device, criterion=criterion)
        log_train.append([loss_train, acc_train])
        log_valid.append([loss_valid, acc_valid])

        torch.save({'epoch': i, 'model_state_dict': model.state_dict(), 'optimizer_state_dict': optimizer.state_dict()}, f'./output/knock89/checkpoint{i + 1}.pt')
        e_time = time.time()

        print(f'epoch: {i + 1}, loss_train: {loss_train:.4f}, accuracy_train: {acc_train:.4f}, loss_valid: {loss_valid:.4f}, accuracy_valid: {acc_valid:.4f}, {(e_time - s_time):.4f}sec')

    return {'train': log_train, 'valid': log_valid}


def plot_log(log, outpath):
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
    plt.savefig(outpath)


if __name__ == "__main__":
    train_data = pd.read_csv('../chapter06/output/train.txt', sep='\t', names=('CATEGORY', 'TITLE'))
    valid_data = pd.read_csv('../chapter06/output/valid.txt', sep='\t', names=('CATEGORY', 'TITLE'))
    test_data = pd.read_csv('../chapter06/output/test.txt', sep='\t', names=('CATEGORY', 'TITLE'))

    X_train_text = train_data["TITLE"]
    X_valid_text = valid_data["TITLE"]
    X_test_text = test_data["TITLE"]

    Y_train = pd.get_dummies(train_data, columns=['CATEGORY'])[['CATEGORY_b', 'CATEGORY_e', 'CATEGORY_t', 'CATEGORY_m']].values
    Y_valid = pd.get_dummies(valid_data, columns=['CATEGORY'])[['CATEGORY_b', 'CATEGORY_e', 'CATEGORY_t', 'CATEGORY_m']].values
    Y_test = pd.get_dummies(test_data, columns=['CATEGORY'])[['CATEGORY_b', 'CATEGORY_e', 'CATEGORY_t', 'CATEGORY_m']].values

    max_len = 20
    tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')

    dataset_train = MakeDataset(X_train_text, Y_train, tokenizer, max_len)
    dataset_valid = MakeDataset(X_valid_text, Y_valid, tokenizer, max_len)
    dataset_test = MakeDataset(X_test_text, Y_test, tokenizer, max_len)

        
    # パラメータの設定
    DROP_RATE = 0.4
    OUTPUT_SIZE = 4
    BATCH_SIZE = 64
    NUM_EPOCHS = 5
    LEARNING_RATE = 2e-5

    # モデル定義
    model = BERTClass(DROP_RATE, OUTPUT_SIZE)
    # 損失関数定義
    criterion = nn.BCEWithLogitsLoss()
    # optimizer定義
    optimizer = torch.optim.AdamW(params=model.parameters(), lr=LEARNING_RATE)
    device = torch.device("cuda:5")
    # モデルの学習
    log = train_model(dataset_train, dataset_valid, model, criterion, optimizer, BATCH_SIZE, NUM_EPOCHS, device=device)
   
    plot_log(log, "./output/knock89_loss_acc.png")
    _, acc_train = calculate_loss_acc(model, dataset_train)
    _, acc_valid = calculate_loss_acc(model, dataset_valid)
    print('正解率')
    print(f'訓練データ：{acc_train:.4f}')
    print(f'開発データ：{acc_valid:.4f}')
    
"""
epoch: 1, loss_train: 0.1138, accuracy_train: 0.9386, loss_valid: 0.1327, accuracy_valid: 0.9123, 181.3608sec
 20%|██████████████▊                                                           | 1/5 [03:01<12:05, 181.36s/it]
epoch: 2, loss_train: 0.0585, accuracy_train: 0.9714, loss_valid: 0.1007, accuracy_valid: 0.9333, 179.4420sec
 40%|█████████████████████████████▌                                            | 2/5 [06:00<09:00, 180.23s/it]
epoch: 3, loss_train: 0.0353, accuracy_train: 0.9849, loss_valid: 0.1004, accuracy_valid: 0.9333, 178.4274sec
 60%|████████████████████████████████████████████▍                             | 3/5 [08:59<05:58, 179.41s/it]
epoch: 4, loss_train: 0.0271, accuracy_train: 0.9880, loss_valid: 0.1018, accuracy_valid: 0.9288, 178.2920sec
 80%|███████████████████████████████████████████████████████████▏              | 4/5 [11:57<02:58, 178.97s/it]
epoch: 5, loss_train: 0.0176, accuracy_train: 0.9938, loss_valid: 0.1040, accuracy_valid: 0.9355, 180.0683sec
100%|██████████████████████████████████████████████████████████████████████████| 5/5 [14:57<00:00, 179.52s/it]
正解率
訓練データ：0.9938
開発データ：0.9355
"""