import torch
import matplotlib.pyplot as plt
import time
from transformers import BertTokenizer, BertModel
import transformers
import numpy as np
from torch.utils.data import Dataset, DataLoader
from torch import nn, optim, cuda


# Dataset
class CreateDataset(Dataset):
    def __init__(self, X, y, tokenizer, max_len):
        self.X = X
        self.y = y
        self.tokenizer = tokenizer
        self.max_len = max_len

    def __len__(self):  # len(Dataset)で返す値を指定
        return len(self.y)

    def __getitem__(self, index):  # Dataset[index]で返す値を指定
        text = self.X[index]
        inputs = self.tokenizer.encode_plus(
            text,
            add_special_tokens=True,
            max_length=self.max_len,
            pad_to_max_length=True
        )
        ids = inputs['input_ids']
        mask = inputs['attention_mask']

        item_dataset = {}
        item_dataset['ids'] = torch.LongTensor(ids)
        item_dataset['mask'] = torch.LongTensor(mask)
        item_dataset['labels'] = torch.Tensor(self.y[index])

        return item_dataset


# Datasetの作成
max_len = 20
tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')
dataset_train = CreateDataset(train_data['TITLE'], y_train, tokenizer, max_len)
dataset_valid = CreateDataset(valid_data['TITLE'], y_valid, tokenizer, max_len)
dataset_test = CreateDataset(test_data['TITLE'], y_test, tokenizer, max_len)

y_train = pd.get_dummies(train_data, columns=['CATEGORY'])[
    ['CATEGORY_b', 'CATEGORY_e', 'CATEGORY_t', 'CATEGORY_m']].values
y_valid = pd.get_dummies(valid_data, columns=['CATEGORY'])[
    ['CATEGORY_b', 'CATEGORY_e', 'CATEGORY_t', 'CATEGORY_m']].values
y_test = pd.get_dummies(test_data, columns=['CATEGORY'])[
    ['CATEGORY_b', 'CATEGORY_e', 'CATEGORY_t', 'CATEGORY_m']].values


class BERTClass(torch.nn.Module):
    '''BERT分類モデルの定義'''
    def __init__(self, drop_rate, otuput_size):
        super().__init__()
        self.bert = BertModel.from_pretrained('bert-base-uncased')
        self.drop = torch.nn.Dropout(drop_rate)
        self.fc = torch.nn.Linear(768, otuput_size)  # BERTの出力に合わせて768次元を指定

    def forward(self, ids, mask):
        _, out = self.bert(ids, attention_mask=mask)
        out = self.fc(self.drop(out))
        return out


# パラメータ
DROP_RATE = 0.4
OUTPUT_SIZE = 4
BATCH_SIZE = 32
NUM_EPOCHS = 4
LEARNING_RATE = 2e-5

# モデル定義
model = BERTClass(DROP_RATE, OUTPUT_SIZE)

# 損失関数
criterion = torch.nn.BCEWithLogitsLoss()

# オプティマイザ
optimizer = torch.optim.AdamW(params=model.parameters(), lr=LEARNING_RATE)

# モデルの学習
log = train_model(dataset_train, dataset_valid, BATCH_SIZE,
                  model, criterion, optimizer, NUM_EPOCHS)

# 損失ログの可視化
plt.plot(np.array(log['train']).T[0], label='train')
plt.plot(np.array(log['valid']).T[0], label='valid')
plt.xlabel('epoch')
plt.ylabel('loss')
plt.legend()
# plt.savefig('./results/output89_loss.png')
plt.show()

# 正解率ログの可視化
plt.plot(np.array(log['train']).T[1], label='train')
plt.plot(np.array(log['valid']).T[1], label='valid')
plt.xlabel('epoch')
plt.ylabel('accuracy')
plt.legend()
# plt.savefig('./results/output89_accuracy.png')
plt.show()

# 正解率の算出
_, acc_train = calculate_loss_and_accuracy(model, dataset_train)
_, acc_test = calculate_loss_and_accuracy(model, dataset_test)
print(f'正解率（学習データ）: {acc_train:.3f}')
print(f'正解率（評価データ）: {acc_test:.3f}')
