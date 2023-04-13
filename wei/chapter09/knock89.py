""'''
89. 事前学習済み言語モデルからの転移学習
事前学習済み言語モデル（例えばBERTなど）を出発点として，
ニュース記事見出しをカテゴリに分類するモデルを構築せよ．
'''
import torch
from torch.utils.data import Dataset, DataLoader
import numpy as np
import pandas as pd
from matplotlib import pyplot as plt
from transformers import BertTokenizer, BertModel
from tqdm import tqdm
import time



# difine dataset
class MakeDataset(Dataset):
    def __init__(self, X, y, tokenizer, max_len):
        self.X = X
        self.y = y
        self.tokenizer = tokenizer
        self.max_len = max_len

    def __len__(self):
        return len(self.y)

    def __getitem__(self, idx):
        text = self.X[idx]
        # tokenizerで前処理をして、指定した最長系列長までpaddingしてから、単語idに変換
        inputs = self.tokenizer.encode_plus(
            text,
            add_special_tokens=True,
            max_length=self.max_len,
            pad_to_max_length=True
        )

        ids = inputs['input_ids']
        mask = inputs['attention_mask']
        return {
            'ids' :torch.LongTensor(ids),
            'mask' :torch.LongTensor(mask),
            'labels' :torch.Tensor(self.y[idx])
        }


# define BERTClass model
class BERTClass(torch.nn.Module):
    def __init__(self, dropout_rate, output_size):
        super().__init__()
        self.bert = BertModel.from_pretrained('bert-base-uncased', return_dict=False)
        self.dropout = torch.nn.Dropout(dropout_rate)
        self.fc = torch.nn.Linear(768, output_size)

    def forward(self, ids, mask):
        _, out = self.bert(ids, attention_mask=mask)
        out = self.fc(self.dropout(out))
        return out

# define cal_loss_acc function
def cal_loss_acc(model, dataset, criterion, device):
    model.eval()
    dataloader = DataLoader(dataset, batch_size=1, shuffle=False)
    loss = 0.0
    total = 0
    crt = 0
    with torch.no_grad():
        for data in dataloader:
            ids = data['ids'].to(device)
            mask = data['mask'].to(device)
            labels = data['labels'].to(device)

            outputs = model.forward(ids, mask)
            if criterion != None:
                loss += criterion(outputs, labels).item()

            # バッチサイズ分の予測ラベル
            pred = torch.argmax(outputs, dim=-1).cpu().numpy()
            labels = torch.argmax(labels, dim=-1).cpu().numpy()
            total += len(labels)
            crt += (pred == labels).sum().item()
        return loss/len(dataset), crt/total


# define train_model
def train_model(dataset_train, dataset_valid, model, criterion, optimizer, batch_size, epoch, device):
    model.to(device)

    dataloader_train = DataLoader(dataset_train, batch_size=batch_size, shuffle=True)
    dataloader_valid = DataLoader(dataset_valid, batch_size=1, shuffle=False)

    train_log, valid_log = [], []
    for i in tqdm(range(epoch)):
        start = time.time()

        model.train()
        for data in dataloader_train:
            optimizer.zero_grad()
            ids = data['ids'].to(device)
            mask = data['mask'].to(device)
            labels = data['labels'].to(device)

            outputs = model.forward(ids, mask)
            loss = criterion(outputs, labels)
            loss.backward()     # backforward
            optimizer.step()     # update weights

        model.eval()
        loss_train, acc_train = cal_loss_acc(model, dataset_train, device, criterion)
        loss_valid, acc_valid = cal_loss_acc(model, dataset_valid, device, criterion)
        train_log.append([loss_train, acc_train])
        valid_log.append([loss_valid, acc_valid])

        # save checkpoints
        model_param_dic = {'epoch': i+1, 'model_state_dict': model.state_dict(), 'optimizer_state_dic': optimizer.state_dict()}
        torch.save(model_param_dic, f'knock89_checkpoint_{i+1}.pth')
        end = time.time()

        print(
            f'epoch: {i + 1}, loss_train: {loss_train:.4f}, accuracy_train: {acc_train:.4f}, loss_valid: {loss_valid:.4f}, accuracy_valid: {acc_valid:.4f}, {(end - start):.4f}sec'
        )

        return {'train': train_log, 'valid': valid_log}


def visualization(log, outpath):
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




if __name__ == '__main__':
    # load data
    train_data = pd.read_csv('../chapter06/train_re.txt', sep='\t', names=['CATEGORY', 'TITLE'])
    valid_data = pd.read_csv('../chapter06/valid_re.txt', sep='\t', names=['CATEGORY', 'TITLE'])
    test_data = pd.read_csv('../chapter06/test_re.txt', sep='\t', names=['CATEGORY', 'TITLE'])

    # make labels
    X_train_text = train_data["TITLE"]
    X_valid_text = valid_data["TITLE"]
    X_test_text = test_data["TITLE"]

    Y_train = pd.get_dummies(train_data, columns=['CATEGORY'])[
        ['CATEGORY_b', 'CATEGORY_e', 'CATEGORY_t', 'CATEGORY_m']].values
    Y_valid = pd.get_dummies(valid_data, columns=['CATEGORY'])[
        ['CATEGORY_b', 'CATEGORY_e', 'CATEGORY_t', 'CATEGORY_m']].values
    Y_test = pd.get_dummies(test_data, columns=['CATEGORY'])[
        ['CATEGORY_b', 'CATEGORY_e', 'CATEGORY_t', 'CATEGORY_m']].values

    max_len = 20
    tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')

    dataset_train = MakeDataset(X_train_text, Y_train, tokenizer, max_len)
    dataset_valid = MakeDataset(X_valid_text, Y_valid, tokenizer, max_len)
    dataset_test = MakeDataset(X_test_text, Y_test, tokenizer, max_len)

    # set parameters
    DROP_RATE = 0.4
    OUTPUT_SIZE = 4
    BATCH_SIZE = 32
    NUM_EPOCHS = 2
    LEARNING_RATE = 1e-3
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')


    # define bert model
    model = BERTClass(dropout_rate=DROP_RATE, output_size=OUTPUT_SIZE)
    criterion = torch.nn.BCEWithLogitsLoss()
    optimizer = torch.optim.AdamW(params=model.parameters(), lr=LEARNING_RATE)

    log = train_model(dataset_train, dataset_valid, model=model,criterion=criterion, optimizer=optimizer, batch_size=BATCH_SIZE,epoch=NUM_EPOCHS,device=device)

    visualization(log,'knock89.png')
    train_loss, train_acc = cal_loss_acc(model, criterion, dataset_train, device)
    valid_loss, valid_acc = cal_loss_acc(model, criterion, dataset_valid, device)
    print(f'train_loss:{train_loss:.4f}, train_acc:{train_acc:.4f}')
    print(f'valid_loss:{valid_loss:.4f}, valid_acc:{valid_acc:.4f}')








