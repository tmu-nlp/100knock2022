import time
import string
import torch
import numpy as np
import pandas as pd
from matplotlib import pyplot as plt
from torch import nn
from gensim.models import KeyedVectors
from torch.utils.data import Dataset
from torch.utils.data import DataLoader
from sklearn.model_selection import train_test_split

df = pd.read_csv('./100knock2022/DUAN/chapter06/newsCorpora.csv', header=None, sep='\t', names=['ID', 'TITLE', 'URL', 'PUBLISHER', 'CATEGORY', 'STORY', 'HOSTNAME', 'TIMESTAMP'])
df = df.loc[df['PUBLISHER'].isin(['Reuters', 'Huffington Post', 'Businessweek', 'Contactmusic.com', 'Daily Mail']), ['TITLE', 'CATEGORY']]
train, valid_test = train_test_split(df, test_size=0.2, shuffle=True, random_state=123, stratify=df['CATEGORY'])
valid, test = train_test_split(valid_test, test_size=0.5, shuffle=True, random_state=123, stratify=valid_test['CATEGORY'])
model = KeyedVectors.load_word2vec_format('./100knock2022/DUAN/chapter08/GoogleNews-vectors-negative300.bin', binary=True)

def transform_w2v(text):
    table = str.maketrans(string.punctuation, ' '*len(string.punctuation))
    words = text.translate(table).split()  
    vec = [model[word] for word in words if word in model]  
    return torch.tensor(sum(vec) / len(vec))

X_train = torch.stack([transform_w2v(text) for text in train['TITLE']])
X_valid = torch.stack([transform_w2v(text) for text in valid['TITLE']])
X_test = torch.stack([transform_w2v(text) for text in test['TITLE']])
category_dict = {'b': 0, 't': 1, 'e':2, 'm':3}
y_train = torch.tensor(train['CATEGORY'].map(lambda x: category_dict[x]).values)
y_valid = torch.tensor(valid['CATEGORY'].map(lambda x: category_dict[x]).values)
y_test = torch.tensor(test['CATEGORY'].map(lambda x: category_dict[x]).values)

class SLPNet(nn.Module):
    def __init__(self, input_size, output_size):
        super().__init__()
        self.fc = nn.Linear(input_size, output_size, bias=False)
        nn.init.normal_(self.fc.weight, 0.0, 1.0) 

    def forward(self, x):
        x = self.fc(x)
        return x

class NewsDataset(Dataset):
    def __init__(self, X, y):  
        self.X = X
        self.y = y

    def __len__(self):  
        return len(self.y)

    def __getitem__(self, idx):  
        return [self.X[idx], self.y[idx]]

dataset_train = NewsDataset(X_train, y_train)
dataset_valid = NewsDataset(X_valid, y_valid)
dataset_test = NewsDataset(X_test, y_test)
dataloader_train = DataLoader(dataset_train, batch_size=1, shuffle=True)
dataloader_valid = DataLoader(dataset_valid, batch_size=len(dataset_valid), shuffle=False)
dataloader_test = DataLoader(dataset_test, batch_size=len(dataset_test), shuffle=False)
model = SLPNet(300, 4)
criterion = nn.CrossEntropyLoss()
optimizer = torch.optim.SGD(model.parameters(), lr=1e-1)

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

def train_model(dataset_train, dataset_valid, batch_size, model, criterion, optimizer, num_epochs):
    dataloader_train = DataLoader(dataset_train, batch_size=batch_size, shuffle=True)
    dataloader_valid = DataLoader(dataset_valid, batch_size=len(dataset_valid), shuffle=False)
    log_train = []
    log_valid = []
    for epoch in range(num_epochs):
        s_time = time.time()
        model.train()
        for inputs, labels in dataloader_train:
            optimizer.zero_grad()
            outputs = model(inputs)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()
        loss_train, acc_train = calculate_loss_and_accuracy(model, criterion, dataloader_train)
        loss_valid, acc_valid = calculate_loss_and_accuracy(model, criterion, dataloader_valid)
        log_train.append([loss_train, acc_train])
        log_valid.append([loss_valid, acc_valid])
        torch.save({'epoch': epoch, 'model_state_dict': model.state_dict(), 'optimizer_state_dict': optimizer.state_dict()}, f'checkpoint{epoch + 1}.pt')          
        e_time = time.time()
    print(f'epoch: {epoch + 1}, loss_train: {loss_train:.4f}, accuracy_train: {acc_train:.4f}, loss_valid: {loss_valid:.4f}, accuracy_valid: {acc_valid:.4f}, {(e_time - s_time):.4f}sec') 
    return {'train': log_train, 'valid': log_valid}

for batch_size in [2 ** i for i in range(11)]:
    print(f'バッチサイズ: {batch_size}')
    log = train_model(dataset_train, dataset_valid, batch_size, model, criterion, optimizer, 1)

'''
バッチサイズ: 1
loss_train: 0.3364, accuracy_train: 0.8816, loss_valid: 0.3602, accuracy_valid: 0.8771, 3.9090sec
バッチサイズ: 2
loss_train: 0.3081, accuracy_train: 0.8938, loss_valid: 0.3329, accuracy_valid: 0.8801, 3.4545sec
バッチサイズ: 4
loss_train: 0.2965, accuracy_train: 0.8977, loss_valid: 0.3188, accuracy_valid: 0.8876, 2.0983sec
バッチサイズ: 8
loss_train: 0.2927, accuracy_train: 0.8990, loss_valid: 0.3151, accuracy_valid: 0.8891, 0.9714sec
バッチサイズ: 16
loss_train: 0.2909, accuracy_train: 0.9005, loss_valid: 0.3136, accuracy_valid: 0.8883, 0.6353sec
バッチサイズ: 32
loss_train: 0.2899, accuracy_train: 0.9009, loss_valid: 0.3125, accuracy_valid: 0.8891, 0.4290sec
バッチサイズ: 64
loss_train: 0.2900, accuracy_train: 0.9012, loss_valid: 0.3124, accuracy_valid: 0.8891, 0.3365sec
バッチサイズ: 128
loss_train: 0.2896, accuracy_train: 0.9011, loss_valid: 0.3121, accuracy_valid: 0.8898, 0.2930sec
バッチサイズ: 256
loss_train: 0.2884, accuracy_train: 0.9012, loss_valid: 0.3119, accuracy_valid: 0.8898, 0.5588sec
バッチサイズ: 512
loss_train: 0.2891, accuracy_train: 0.9012, loss_valid: 0.3119, accuracy_valid: 0.8898, 0.2679sec
バッチサイズ: 1024
loss_train: 0.2916, accuracy_train: 0.9013, loss_valid: 0.3118, accuracy_valid: 0.8898, 0.2488sec
'''