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

model = SLPNet(300, 4)
criterion = nn.CrossEntropyLoss()
optimizer = torch.optim.SGD(model.parameters(), lr=1e-1)

log_train = []
log_valid = []
epochs = 10
for epoch in range(epochs):
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
    print(f'epoch: {epoch + 1}, loss_train: {loss_train:.4f}, accuracy_train: {acc_train:.4f}, loss_valid: {loss_valid:.4f}, accuracy_valid: {acc_valid:.4f}')  

'''
epoch: 1, loss_train: 0.3342, accuracy_train: 0.8837, loss_valid: 0.3688, accuracy_valid: 0.8658
epoch: 2, loss_train: 0.2884, accuracy_train: 0.9010, loss_valid: 0.3233, accuracy_valid: 0.8831
epoch: 3, loss_train: 0.2687, accuracy_train: 0.9070, loss_valid: 0.3016, accuracy_valid: 0.8906
epoch: 4, loss_train: 0.2651, accuracy_train: 0.9117, loss_valid: 0.3053, accuracy_valid: 0.8921
epoch: 5, loss_train: 0.2516, accuracy_train: 0.9137, loss_valid: 0.2885, accuracy_valid: 0.8981
epoch: 6, loss_train: 0.2495, accuracy_train: 0.9165, loss_valid: 0.2890, accuracy_valid: 0.9055
epoch: 7, loss_train: 0.2379, accuracy_train: 0.9198, loss_valid: 0.2774, accuracy_valid: 0.8996
epoch: 8, loss_train: 0.2396, accuracy_train: 0.9191, loss_valid: 0.2813, accuracy_valid: 0.9003
epoch: 9, loss_train: 0.2313, accuracy_train: 0.9229, loss_valid: 0.2727, accuracy_valid: 0.9040
epoch: 10, loss_train: 0.2310, accuracy_train: 0.9209, loss_valid: 0.2732, accuracy_valid: 0.9010
'''