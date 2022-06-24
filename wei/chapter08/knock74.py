'''
74. 正解率の計測
knock73で求めた行列を用いて、学習データと評価データの事例を分類した時、その正解率をそれぞれ求める。
'''
from knock71 import sglNN
import torch
from torch.utils.data import TensorDataset,DataLoader


def cal_acc(model, loader):
    model.eval()
    total = 0
    correct = 0
    with torch.no_grad():
        for inputs, labels in loader:
            outputs = model(inputs)
            pred = torch.argmax(outputs, dim=-1)
            total += len(inputs)
            correct += (pred == labels).sum().item()
    return correct/total


X_train = torch.load('X_train.pt')
X_valid = torch.load('X_valid.pt')
X_test = torch.load('X_test.pt')
y_train = torch.load('y_train.pt')
y_valid = torch.load('y_valid.pt')
y_test = torch.load('y_test.pt')

my_data_train = TensorDataset(X_train, y_train)
d_loader_train = DataLoader(my_data_train, batch_size=1, shuffle=True, drop_last=False)  # epoch毎にshuffled,
my_data_valid = TensorDataset(X_valid, y_valid)
d_loader_valid = DataLoader(my_data_valid, batch_size=len(my_data_valid), shuffle=False, drop_last=False)
my_data_test = TensorDataset(X_test, y_test)
d_loader_test = DataLoader(my_data_test, batch_size=len(my_data_test), shuffle=False, drop_last=False)

my_nn = sglNN(300, 4)
model_path = 'knock73_model.bin'
my_nn.load_state_dict(torch.load(model_path))   # load trained model

acc_train = cal_acc(my_nn, d_loader_train)
acc_valid = cal_acc(my_nn, d_loader_valid)
acc_test = cal_acc(my_nn, d_loader_test)

print(f'acc on train: {acc_train:.4f}')
print(f'acc on valid: {acc_valid:.4f}')
print(f'acc on test: {acc_test:.4f}')


'''
acc on train: 0.9081
acc on valid: 0.8853
acc on test: 0.8778
'''