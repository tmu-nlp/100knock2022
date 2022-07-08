import time
import torch
import optuna
import string
import pandas as pd
import numpy as np
from torch import nn
from torch import optim
from torch.nn import functional as F
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

class textCNN(nn.Module):
    def __init__(self, vocab_size, emb_size, padding_idx, output_size, out_channels, conv_params, drop_rate, emb_weights=None):
        super().__init__()
        if emb_weights != None:  
            self.emb = nn.Embedding.from_pretrained(emb_weights, padding_idx=padding_idx)
        else:
            self.emb = nn.Embedding(vocab_size, emb_size, padding_idx=padding_idx)
        self.convs = nn.ModuleList([nn.Conv2d(1, out_channels, (kernel_height, emb_size), padding=(padding, 0)) for kernel_height, padding in conv_params])
        self.drop = nn.Dropout(drop_rate)
        self.fc = nn.Linear(len(conv_params) * out_channels, output_size)

    def forward(self, x):
        emb = self.emb(x).unsqueeze(1)
        conv = [F.relu(conv(emb)).squeeze(3) for i, conv in enumerate(self.convs)]
        max_pool = [F.max_pool1d(i, i.size(2)) for i in conv]
        max_pool_cat = torch.cat(max_pool, 1)
        out = self.fc(self.drop(max_pool_cat.squeeze(2)))
        return out

def objective(trial):
    emb_size = int(trial.suggest_discrete_uniform('emb_size', 100, 400, 100))
    out_channels = int(trial.suggest_discrete_uniform('out_channels', 50, 200, 50))
    drop_rate = trial.suggest_discrete_uniform('drop_rate', 0.0, 0.5, 0.1)
    learning_rate = trial.suggest_loguniform('learning_rate', 5e-4, 5e-2)
    momentum = trial.suggest_discrete_uniform('momentum', 0.5, 0.9, 0.1)
    batch_size = int(trial.suggest_discrete_uniform('batch_size', 16, 128, 16))
    VOCAB_SIZE = len(set(word2id.values())) + 1
    PADDING_IDX = len(set(word2id.values()))
    OUTPUT_SIZE = 4
    CONV_PARAMS = [[2, 0], [3, 1], [4, 2]]
    NUM_EPOCHS = 30
    model = textCNN(VOCAB_SIZE, EMB_SIZE, PADDING_IDX, OUTPUT_SIZE, out_channels, CONV_PARAMS, drop_rate, emb_weights=weights)
    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.SGD(model.parameters(), lr=learning_rate, momentum=momentum)
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    log = train_model(dataset_train, dataset_valid, batch_size, model, criterion, optimizer, NUM_EPOCHS, collate_fn=Padsequence(PADDING_IDX), device=device)
    loss_valid, _ = calculate_loss_and_accuracy(model, dataset_valid, device, criterion=criterion) 
    return loss_valid 
'''
study = optuna.create_study()
study.optimize(objective, timeout=600)
print('Best trial:')
trial = study.best_trial
print('  Value: {:.3f}'.format(trial.value))
print('  Params: ')
for key, value in trial.params.items():
    print('    {}: {}'.format(key, value))
'''
VOCAB_SIZE = len(set(word2id.values())) + 1
EMB_SIZE = 300
PADDING_IDX = len(set(word2id.values()))
OUTPUT_SIZE = 4
OUT_CHANNELS = 200
CONV_PARAMS = [[2, 0], [3, 1], [4, 2]]
DROP_RATE = 0.0
LEARNING_RATE = 6e-3
BATCH_SIZE = 128
NUM_EPOCHS = 30

model = textCNN(VOCAB_SIZE, EMB_SIZE, PADDING_IDX, OUTPUT_SIZE, OUT_CHANNELS, CONV_PARAMS, DROP_RATE, emb_weights=weights)
print(model)
criterion = nn.CrossEntropyLoss()
optimizer = torch.optim.SGD(model.parameters(), lr=LEARNING_RATE, momentum=0.9)
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
log = train_model(dataset_train, dataset_valid, BATCH_SIZE, model, criterion, optimizer, NUM_EPOCHS, collate_fn=Padsequence(PADDING_IDX), device=device)
visualize_logs(log)
n, acc_train = calculate_loss_and_accuracy(model, dataset_train, device)
t, acc_test = calculate_loss_and_accuracy(model, dataset_test, device)
print(f'正解率（学習データ）：{acc_train:.3f}')
print(f'正解率（評価データ）：{acc_test:.3f}')

'''
poch: 1, loss_train: 1.1165, accuracy_train: 0.5238, loss_valid: 1.1179, accuracy_valid: 0.5217, 44.8156sec
epoch: 2, loss_train: 1.0646, accuracy_train: 0.5963, loss_valid: 1.0735, accuracy_valid: 0.5682, 40.6319sec
epoch: 3, loss_train: 1.0104, accuracy_train: 0.6212, loss_valid: 1.0300, accuracy_valid: 0.5952, 35.5207sec
epoch: 4, loss_train: 0.9529, accuracy_train: 0.6596, loss_valid: 0.9836, accuracy_valid: 0.6199, 35.0890sec
epoch: 5, loss_train: 0.8965, accuracy_train: 0.6855, loss_valid: 0.9347, accuracy_valid: 0.6574, 35.3695sec
epoch: 6, loss_train: 0.8434, accuracy_train: 0.7101, loss_valid: 0.8980, accuracy_valid: 0.6754, 38.5905sec
epoch: 7, loss_train: 0.7958, accuracy_train: 0.7257, loss_valid: 0.8669, accuracy_valid: 0.6882, 38.7636sec
epoch: 8, loss_train: 0.7543, accuracy_train: 0.7387, loss_valid: 0.8346, accuracy_valid: 0.6979, 34.8583sec
epoch: 9, loss_train: 0.7167, accuracy_train: 0.7539, loss_valid: 0.8180, accuracy_valid: 0.7031, 39.0363sec
epoch: 10, loss_train: 0.6805, accuracy_train: 0.7647, loss_valid: 0.7940, accuracy_valid: 0.7099, 39.0452sec
epoch: 11, loss_train: 0.6512, accuracy_train: 0.7679, loss_valid: 0.7780, accuracy_valid: 0.7144, 35.0624sec
epoch: 12, loss_train: 0.6228, accuracy_train: 0.7801, loss_valid: 0.7578, accuracy_valid: 0.7249, 34.9454sec
epoch: 13, loss_train: 0.5957, accuracy_train: 0.7909, loss_valid: 0.7428, accuracy_valid: 0.7264, 46.3250sec
epoch: 14, loss_train: 0.5731, accuracy_train: 0.7970, loss_valid: 0.7270, accuracy_valid: 0.7309, 38.2742sec
epoch: 15, loss_train: 0.5533, accuracy_train: 0.8116, loss_valid: 0.7232, accuracy_valid: 0.7429, 47.4310sec
epoch: 16, loss_train: 0.5339, accuracy_train: 0.8135, loss_valid: 0.7083, accuracy_valid: 0.7391, 46.5016sec
epoch: 17, loss_train: 0.5176, accuracy_train: 0.8222, loss_valid: 0.7009, accuracy_valid: 0.7466, 37.4299sec
epoch: 18, loss_train: 0.5043, accuracy_train: 0.8316, loss_valid: 0.6954, accuracy_valid: 0.7511, 40.8478sec
epoch: 19, loss_train: 0.4918, accuracy_train: 0.8346, loss_valid: 0.6880, accuracy_valid: 0.7496, 37.2433sec
epoch: 20, loss_train: 0.4822, accuracy_train: 0.8364, loss_valid: 0.6791, accuracy_valid: 0.7541, 35.7414sec
epoch: 21, loss_train: 0.4737, accuracy_train: 0.8469, loss_valid: 0.6806, accuracy_valid: 0.7571, 35.5754sec
epoch: 22, loss_train: 0.4667, accuracy_train: 0.8504, loss_valid: 0.6761, accuracy_valid: 0.7586, 35.9072sec
epoch: 23, loss_train: 0.4613, accuracy_train: 0.8507, loss_valid: 0.6725, accuracy_valid: 0.7586, 36.4273sec
epoch: 24, loss_train: 0.4573, accuracy_train: 0.8544, loss_valid: 0.6725, accuracy_valid: 0.7594, 36.6145sec
epoch: 25, loss_train: 0.4541, accuracy_train: 0.8540, loss_valid: 0.6683, accuracy_valid: 0.7601, 36.6371sec
epoch: 26, loss_train: 0.4518, accuracy_train: 0.8538, loss_valid: 0.6666, accuracy_valid: 0.7579, 35.8309sec
epoch: 27, loss_train: 0.4504, accuracy_train: 0.8544, loss_valid: 0.6658, accuracy_valid: 0.7594, 36.1898sec
epoch: 28, loss_train: 0.4495, accuracy_train: 0.8549, loss_valid: 0.6657, accuracy_valid: 0.7586, 39.0410sec
epoch: 29, loss_train: 0.4491, accuracy_train: 0.8549, loss_valid: 0.6654, accuracy_valid: 0.7586, 35.8920sec
epoch: 30, loss_train: 0.4490, accuracy_train: 0.8549, loss_valid: 0.6653, accuracy_valid: 0.7586, 42.5632sec
正解率（学習データ）：0.855
正解率（評価データ）：0.778
'''