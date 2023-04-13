import time
import torch
import string
import pandas as pd
import numpy as np
from torch import nn
from torch import optim
from collections import defaultdict
from matplotlib import pyplot as plt
from torch.utils.data import Dataset
from torch.utils.data import DataLoader
from sklearn.model_selection import train_test_split
from gensim.models import KeyedVectors

def calculate_loss_and_accuracy(model, dataset, device = None, criterion = None):
    dataloader = DataLoader(dataset, batch_size = 1, shuffle = False)
    loss = 0.0
    total = 0
    correct = 0
    with torch.no_grad():
        for data in dataloader:
            inputs = data['inputs'].to(device)
            labels = data['labels'].to(device)
            outputs = model(inputs)
            if criterion != None:
                loss += criterion(outputs, labels).item()
            pred = torch.argmax(outputs, dim=-1)
            total += len(inputs)
            correct += (pred == labels).sum().item()
    return loss / len(dataset), correct / total

def visualize_logs(log):
    fig, ax = plt.subplots(1, 2, figsize = (15, 5))
    ax[0].plot(np.array(log['train']).T[0], label = 'train')
    ax[0].plot(np.array(log['valid']).T[0], label = 'valid')
    ax[0].set_xlabel('epoch')
    ax[0].set_ylabel('loss')
    ax[0].legend()
    ax[1].plot(np.array(log['train']).T[1], label = 'train')
    ax[1].plot(np.array(log['valid']).T[1], label = 'valid')
    ax[1].set_xlabel('epoch')
    ax[1].set_ylabel('accuracy')
    ax[1].legend()
    plt.show()

def train_model(dataset_train, dataset_valid, batch_size, model, criterion, optimizer, num_epochs, collate_fn = None, device = None):
    model.to(device)
    dataloader_train = DataLoader(dataset_train, batch_size = batch_size, shuffle = True, collate_fn = collate_fn)
    dataloader_valid = DataLoader(dataset_valid, batch_size = 1, shuffle = False)
    scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, num_epochs, eta_min = 1e-5, last_epoch = -1)
    
    log_train = []
    log_valid = []
    for epoch in range(num_epochs):
        s_time = time.time()
        model.train()
        for data in dataloader_train:
            optimizer.zero_grad()
            inputs = data['inputs'].to(device)
            labels = data['labels'].to(device)
            outputs = model(inputs)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()
        model.eval()

        loss_train, acc_train = calculate_loss_and_accuracy(model, dataset_train, device, criterion = criterion)
        loss_valid, acc_valid = calculate_loss_and_accuracy(model, dataset_valid, device, criterion = criterion)
        log_train.append([loss_train, acc_train])
        log_valid.append([loss_valid, acc_valid])

        torch.save({'epoch': epoch, 'model_state_dict': model.state_dict(), 'optimizer_state_dict': optimizer.state_dict()}, f'checkpoint{epoch + 1}.pt')
        e_time = time.time()
        print(f'epoch: {epoch + 1}, loss_train: {loss_train:.4f}, accuracy_train: {acc_train:.4f}, loss_valid: {loss_valid:.4f}, accuracy_valid: {acc_valid:.4f}, {(e_time - s_time):.4f}sec') 
        if epoch > 2 and log_valid[epoch - 3][0] <= log_valid[epoch - 2][0] <= log_valid[epoch - 1][0] <= log_valid[epoch][0]:
            break
        scheduler.step()
    return {'train': log_train, 'valid': log_valid}

df = pd.read_csv('./100knock2022/DUAN/chapter06/newsCorpora.csv', header = None, sep = '\t', names = ['ID', 'TITLE', 'URL', 'PUBLISHER', 'CATEGORY', 'STORY', 'HOSTNAME', 'TIMESTAMP'])
df = df.loc[df['PUBLISHER'].isin(['Reuters', 'Huffington Post', 'Businessweek', 'Contactmusic.com', 'Daily Mail']), ['TITLE', 'CATEGORY']]
train, valid_test = train_test_split(df, test_size = 0.2, shuffle = True, random_state = 123, stratify = df['CATEGORY'])
valid, test = train_test_split(valid_test, test_size = 0.5, shuffle = True, random_state = 123, stratify = valid_test['CATEGORY'])
train.reset_index(drop = True, inplace = True)
valid.reset_index(drop = True, inplace = True)
test.reset_index(drop = True, inplace = True)

d = defaultdict(int)
table = str.maketrans(string.punctuation, ' '*len(string.punctuation))  
for text in train['TITLE']:
    for word in text.translate(table).split():
        d[word] += 1
d = sorted(d.items(), key=lambda x:x[1], reverse=True)

word2id = {word: i + 1 for i, (word, cnt) in enumerate(d) if cnt > 1} 

def tokenizer(text, word2id = word2id, unk = 0):
    table = str.maketrans(string.punctuation, ' '*len(string.punctuation))
    return [word2id.get(word, unk) for word in text.translate(table).split()]

class RNN(nn.Module):
    def __init__(self, vocab_size, emb_size, padding_idx, output_size, hidden_size, num_layers, emb_weights=None, bidirectional=False):
        super().__init__()
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        self.num_directions = bidirectional + 1 
        if emb_weights != None:  
            self.emb = nn.Embedding.from_pretrained(emb_weights, padding_idx=padding_idx)
        else:
            self.emb = nn.Embedding(vocab_size, emb_size, padding_idx=padding_idx)
        self.rnn = nn.RNN(emb_size, hidden_size, num_layers, nonlinearity='tanh', bidirectional=bidirectional, batch_first=True)
        self.fc = nn.Linear(hidden_size * self.num_directions, output_size)

    def forward(self, x):
        self.batch_size = x.size()[0]
        hidden = self.init_hidden()  
        emb = self.emb(x)
        out, hidden = self.rnn(emb, hidden)
        out = self.fc(out[:, -1, :])
        return out

    def init_hidden(self):
        hidden = torch.zeros(4, self.batch_size, self.hidden_size)
        return hidden

class CreateDataset(Dataset):
    def __init__(self, X, y, tokenizer):
        self.X = X
        self.y = y
        self.tokenizer = tokenizer

    def __len__(self): 
        return len(self.y)

    def __getitem__(self, index): 
        text = self.X[index]
        inputs = self.tokenizer(text)
        return {'inputs': torch.tensor(inputs, dtype = torch.int64),'labels': torch.tensor(self.y[index], dtype = torch.int64)}

category_dict = {'b': 0, 't': 1, 'e':2, 'm':3}
y_train = train['CATEGORY'].map(lambda x: category_dict[x]).values
y_valid = valid['CATEGORY'].map(lambda x: category_dict[x]).values
y_test = test['CATEGORY'].map(lambda x: category_dict[x]).values
dataset_train = CreateDataset(train['TITLE'], y_train, tokenizer)
dataset_valid = CreateDataset(valid['TITLE'], y_valid, tokenizer)
dataset_test = CreateDataset(test['TITLE'], y_test, tokenizer)

class Padsequence():
    def __init__(self, padding_idx):
        self.padding_idx = padding_idx

    def __call__(self, batch):
        sorted_batch = sorted(batch, key=lambda x: x['inputs'].shape[0], reverse=True)
        sequences = [x['inputs'] for x in sorted_batch]
        sequences_padded = torch.nn.utils.rnn.pad_sequence(sequences, batch_first=True, padding_value=self.padding_idx)
        labels = torch.LongTensor([x['labels'] for x in sorted_batch])
        return {'inputs': sequences_padded, 'labels': labels}

model = KeyedVectors.load_word2vec_format('./100knock2022/DUAN/chapter07/GoogleNews-vectors-negative300.bin', binary=True)
VOCAB_SIZE = len(set(word2id.values())) + 1
EMB_SIZE = 300
weights = np.zeros((VOCAB_SIZE, EMB_SIZE))
words_in_pretrained = 0
for i, word in enumerate(word2id.keys()):
    try:
        weights[i] = model[word]
        words_in_pretrained += 1
    except KeyError:
        weights[i] = np.random.normal(scale=0.4, size=(EMB_SIZE))
weights = torch.from_numpy(weights.astype((np.float32)))

VOCAB_SIZE = len(set(word2id.values())) + 1
EMB_SIZE = 300
PADDING_IDX = len(set(word2id.values()))
OUTPUT_SIZE = 4
HIDDEN_SIZE = 50
NUM_LAYERS = 2
LEARNING_RATE = 5e-2
BATCH_SIZE = 32
NUM_EPOCHS = 10
model = RNN(VOCAB_SIZE, EMB_SIZE, PADDING_IDX, OUTPUT_SIZE, HIDDEN_SIZE, NUM_LAYERS, emb_weights=weights, bidirectional=True)
criterion = nn.CrossEntropyLoss()
optimizer = torch.optim.SGD(model.parameters(), lr=LEARNING_RATE)
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
log = train_model(dataset_train, dataset_valid, BATCH_SIZE, model, criterion, optimizer, NUM_EPOCHS, collate_fn=Padsequence(PADDING_IDX), device=device)

visualize_logs(log)
n, acc_train = calculate_loss_and_accuracy(model, dataset_train, device)
t, acc_test = calculate_loss_and_accuracy(model, dataset_test, device)
print(f'正解率（学習データ）：{acc_train:.3f}')
print(f'正解率（評価データ）：{acc_test:.3f}')

'''
epoch: 1, loss_train: 1.1610, accuracy_train: 0.4556, loss_valid: 1.1707, accuracy_valid: 0.4498, 18.0242sec
epoch: 2, loss_train: 1.0643, accuracy_train: 0.5861, loss_valid: 1.1009, accuracy_valid: 0.5540, 17.2323sec
epoch: 3, loss_train: 1.0577, accuracy_train: 0.5899, loss_valid: 1.1123, accuracy_valid: 0.5547, 16.8394sec
epoch: 4, loss_train: 1.0668, accuracy_train: 0.5774, loss_valid: 1.0979, accuracy_valid: 0.5412, 20.7785sec
epoch: 5, loss_train: 1.0585, accuracy_train: 0.5852, loss_valid: 1.1065, accuracy_valid: 0.5585, 22.2308sec
epoch: 6, loss_train: 1.0454, accuracy_train: 0.5992, loss_valid: 1.1007, accuracy_valid: 0.5682, 22.4325sec
epoch: 7, loss_train: 1.0542, accuracy_train: 0.5914, loss_valid: 1.1217, accuracy_valid: 0.5690, 21.4119sec
epoch: 8, loss_train: 1.0148, accuracy_train: 0.6157, loss_valid: 1.0711, accuracy_valid: 0.5937, 20.6179sec
epoch: 9, loss_train: 1.0143, accuracy_train: 0.6160, loss_valid: 1.0824, accuracy_valid: 0.5855, 22.2127sec
epoch: 10, loss_train: 0.9843, accuracy_train: 0.6302, loss_valid: 1.0480, accuracy_valid: 0.6072, 21.3110sec
正解率（学習データ）：0.630
正解率（評価データ）：0.623
'''