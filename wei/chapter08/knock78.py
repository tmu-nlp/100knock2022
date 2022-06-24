'''
78. GPU上での学習
'''
'''
77.MiniBatch
batch_sizeごとに損失・勾配を計算し、行列Wの値を更新(minibatch normalization)
batch_sizeを1,2,4,8…と変化させながら、1エポックの学習に要する時間を比較
'''
from knock75 import cal_loss_acc
from knock71 import sglNN
import datetime
from torch.utils.data import DataLoader, TensorDataset
import torch


def train_model(data_train, data_valid, batch_size, num_epochs, model, optimizer, criterion, device=None):
    model.to(device)

    log_train = []
    log_valid = []

    for epoch in range(1, num_epochs+1):
        start = datetime.datetime.now()

        model.train()
        for inputs, labels in data_train:
            optimizer.zero_grad()

            outputs = model(inputs)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()

        loss_train, acc_train = cal_loss_acc(model, criterion, data_train)
        loss_valid, acc_valid = cal_loss_acc(model, criterion, data_valid)
        log_train.append([loss_train, acc_train])
        log_valid.append([loss_valid, acc_valid])

        # save checkpoints
        # model_param_dic = {'epoch': epoch, 'model_state_dict': model.state_dict(), 'optimizer_state_dic': optimizer.state_dict()}
        # torch.save(model_param_dic, f'knock78_checkpoint_{batch_size}_{epoch}.pth')

        end = datetime.datetime.now()
        time_used = end - start


        print(f'epoch:{epoch}, loss_train:{loss_train:.4f}, acc_train:{acc_train:.4f}, loss_valid:{loss_valid:.4f}, acc_valid:{acc_valid:.4f}, time_used:{time_used}')

    return {'train_log': log_train, 'valid_log': log_valid}


if __name__ == '__main__'    :
    X_train = torch.load('X_train.pt', map_location=lambda storage, loc: storage.cuda())
    X_valid = torch.load('X_valid.pt',map_location=lambda storage, loc: storage.cuda())
    y_train = torch.load('y_train.pt', map_location=lambda storage, loc: storage.cuda())
    y_valid = torch.load('y_valid.pt', map_location=lambda storage, loc: storage.cuda())

    for batch_size in [2**i for i in range(15)]:
        # batch_size毎に、学習に使うデータ量が違うため、DataLoaderを作る時に決める
        my_data_train = TensorDataset(X_train, y_train)
        d_loader_train = DataLoader(my_data_train, batch_size=batch_size, shuffle=True, drop_last=False)  # epoch毎にshuffled
        my_data_valid = TensorDataset(X_valid, y_valid)
        d_loader_valid = DataLoader(my_data_valid, batch_size=len(my_data_valid), shuffle=False, drop_last=False)

        my_nn = sglNN(300, 4)
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        criterion = torch.nn.CrossEntropyLoss()
        optimizer = torch.optim.SGD(my_nn.parameters(), lr=1e-1)

        print(f'batch size: {batch_size}')
        log = train_model(d_loader_train, d_loader_valid, batch_size=batch_size, num_epochs=1, model=my_nn, criterion=criterion, optimizer=optimizer, device=device)
'''
batch size: 1
epoch:1, loss_train:0.3804, acc_train:0.8695, loss_valid:0.4221, acc_valid:0.8508, time_used:0:00:14.991142
batch size: 2
epoch:1, loss_train:0.4395, acc_train:0.8480, loss_valid:0.4471, acc_valid:0.8456, time_used:0:00:07.230898
batch size: 4
epoch:1, loss_train:0.5502, acc_train:0.8009, loss_valid:0.5622, acc_valid:0.7961, time_used:0:00:03.482818
batch size: 8
epoch:1, loss_train:0.6541, acc_train:0.7691, loss_valid:0.6942, acc_valid:0.7586, time_used:0:00:02.218287
batch size: 16
epoch:1, loss_train:0.9375, acc_train:0.6989, loss_valid:0.9463, acc_valid:0.6859, time_used:0:00:01.103136
batch size: 32
epoch:1, loss_train:0.9384, acc_train:0.6566, loss_valid:0.9388, acc_valid:0.6364, time_used:0:00:00.767593
batch size: 64
epoch:1, loss_train:1.0221, acc_train:0.6275, loss_valid:1.0354, acc_valid:0.6139, time_used:0:00:00.567842
batch size: 128
epoch:1, loss_train:1.4193, acc_train:0.4268, loss_valid:1.4020, acc_valid:0.4198, time_used:0:00:00.442641
batch size: 256
epoch:1, loss_train:1.4823, acc_train:0.3388, loss_valid:1.4523, acc_valid:0.3328, time_used:0:00:00.340311
batch size: 512
epoch:1, loss_train:1.3257, acc_train:0.4448, loss_valid:1.3164, acc_valid:0.4558, time_used:0:00:00.292532
batch size: 1024
epoch:1, loss_train:1.7606, acc_train:0.2830, loss_valid:1.7725, acc_valid:0.2954, time_used:0:00:00.319147
batch size: 2048
epoch:1, loss_train:2.3696, acc_train:0.1735, loss_valid:2.3692, acc_valid:0.1777, time_used:0:00:00.328087
batch size: 4096
epoch:1, loss_train:1.8801, acc_train:0.2058, loss_valid:1.8839, acc_valid:0.1987, time_used:0:00:00.347037
batch size: 8192
epoch:1, loss_train:1.7946, acc_train:0.2814, loss_valid:1.7753, acc_valid:0.2916, time_used:0:00:00.341088
batch size: 16384
epoch:1, loss_train:1.4251, acc_train:0.4356, loss_valid:1.4717, acc_valid:0.3996, time_used:0:00:00.332114
'''