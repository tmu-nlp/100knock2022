'''
76. checkpoint
各エポックのパラメータ更新が完了するたびに、
チェックポイント(学習途中のパラメータの値や最適化アルゴリズムの内部状態)をファイルに書き出す
'''

from knock71 import sglNN
import torch
from torch.utils.data import TensorDataset, DataLoader
from knock75 import cal_loss_acc

X_train = torch.load('X_train.pt')
X_valid = torch.load('X_valid.pt')
X_test = torch.load('X_test.pt')
y_train = torch.load('y_train.pt')
y_valid = torch.load('y_valid.pt')
y_test = torch.load('y_test.pt')

my_data_train = TensorDataset(X_train, y_train)
d_loader_train = DataLoader(my_data_train, batch_size=1, shuffle=True, drop_last=False)  # epoch毎にshuffled
my_data_valid = TensorDataset(X_valid, y_valid)
d_loader_valid = DataLoader(my_data_valid, batch_size=len(my_data_valid), shuffle=False, drop_last=False)
my_data_test = TensorDataset(X_test, y_test)
d_loader_test = DataLoader(my_data_test, batch_size=len(my_data_test), shuffle=False, drop_last=False)

my_nn = sglNN(300, 4)
criterion = torch.nn.CrossEntropyLoss()
optimizer = torch.optim.SGD(my_nn.parameters(), lr=1e-1)

num_epochs = 10
for epoch in range(1, num_epochs+1):
    # training mode
    my_nn.train()
    for inputs, labels in d_loader_train:
        # 勾配をゼロで初期化
        optimizer.zero_grad()

        outputs = my_nn(inputs)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()

    loss_train, acc_train = cal_loss_acc(my_nn, criterion, d_loader_train)
    loss_valid, acc_valid = cal_loss_acc(my_nn, criterion, d_loader_valid)
    print(
        f'epoch:{epoch}, loss_train:{loss_train:.4f}, acc_train:{acc_train:.4f}, loss_valid:{loss_valid:.4f}, acc_valid:{acc_valid:.4f}'
    )

    # save checkpoints
    # torch.save(dict_obj, dir)->dict_objにはエポックごとにmodel、optimizerのパラメータを保存
    model_param_dic = {'epoch': epoch, 'model_state_dict': my_nn.state_dict(), 'optimizer_state_dict': optimizer.state_dict()}
    torch.save(model_param_dic, f'knock76_checkpoint_{epoch}.pth')


'''
epoch:1, loss_train:0.3886, acc_train:0.8682, loss_valid:0.4075, acc_valid:0.8493
epoch:2, loss_train:0.3363, acc_train:0.8855, loss_valid:0.3643, acc_valid:0.8703
epoch:3, loss_train:0.3158, acc_train:0.8928, loss_valid:0.3519, acc_valid:0.8756
epoch:4, loss_train:0.3012, acc_train:0.8967, loss_valid:0.3397, acc_valid:0.8778
epoch:5, loss_train:0.2925, acc_train:0.8991, loss_valid:0.3310, acc_valid:0.8846
epoch:6, loss_train:0.2847, acc_train:0.9028, loss_valid:0.3267, acc_valid:0.8831
epoch:7, loss_train:0.2825, acc_train:0.9033, loss_valid:0.3284, acc_valid:0.8808
epoch:8, loss_train:0.2756, acc_train:0.9055, loss_valid:0.3245, acc_valid:0.8846
epoch:9, loss_train:0.2720, acc_train:0.9067, loss_valid:0.3237, acc_valid:0.8853
epoch:10, loss_train:0.2746, acc_train:0.9055, loss_valid:0.3334, acc_valid:0.8846
'''