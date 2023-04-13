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
import time
from torch.utils.data import DataLoader, TensorDataset
import torch


def train_model(data_train, data_valid, batch_size, num_epochs, model, optimizer, criterion, device=None):
    model.to(device)

    log_train = []
    log_valid = []

    for epoch in range(1, num_epochs+1):
        start = time.time()

        model.train()
        for inputs, labels in data_train:
            optimizer.zero_grad()

            outputs = model.forward(inputs)
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

        end = time.time()
        time_used = end - start


        print(f'epoch:{epoch}, loss_train:{loss_train:.4f}, acc_train:{acc_train:.4f}, loss_valid:{loss_valid:.4f}, acc_valid:{acc_valid:.4f}, time_used:{time_used:.4f}')

    return {'train_log': log_train, 'valid_log': log_valid}


if __name__ == '__main__'    :
    X_train = torch.load('X_train.pt', map_location=lambda storage, loc: storage.cuda())
    X_valid = torch.load('X_valid.pt',map_location=lambda storage, loc: storage.cuda())
    y_train = torch.load('y_train.pt', map_location=lambda storage, loc: storage.cuda())
    y_valid = torch.load('y_valid.pt', map_location=lambda storage, loc: storage.cuda())
    my_nn = sglNN(300, 4)
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    criterion = torch.nn.CrossEntropyLoss()
    optimizer = torch.optim.SGD(my_nn.parameters(), lr=1e-1)

    for batch_size in [2**i for i in range(15)]:
        # batch_size毎に、学習に使うデータ量が違うため、DataLoaderを作る時に決める
        my_data_train = TensorDataset(X_train, y_train)
        d_loader_train = DataLoader(my_data_train, batch_size=batch_size, shuffle=True, drop_last=False)  # epoch毎にshuffled
        my_data_valid = TensorDataset(X_valid, y_valid)
        d_loader_valid = DataLoader(my_data_valid, batch_size=len(my_data_valid), shuffle=False, drop_last=False)



        print(f'batch size: {batch_size}')
        train_model(d_loader_train, d_loader_valid, batch_size=batch_size, num_epochs=1, model=my_nn, criterion=criterion, optimizer=optimizer, device=device)
'''
batch size: 1
epoch:1, loss_train:0.3802, acc_train:0.8716, loss_valid:0.4035, acc_valid:0.8583, time_used:8.6667
batch size: 2
epoch:1, loss_train:0.3516, acc_train:0.8800, loss_valid:0.3788, acc_valid:0.8621, time_used:4.1802
batch size: 4
epoch:1, loss_train:0.3422, acc_train:0.8828, loss_valid:0.3713, acc_valid:0.8688, time_used:2.2942
batch size: 8
epoch:1, loss_train:0.3374, acc_train:0.8859, loss_valid:0.3663, acc_valid:0.8733, time_used:1.2301
batch size: 16
epoch:1, loss_train:0.3355, acc_train:0.8861, loss_valid:0.3645, acc_valid:0.8771, time_used:0.6881
batch size: 32
epoch:1, loss_train:0.3341, acc_train:0.8868, loss_valid:0.3636, acc_valid:0.8786, time_used:0.4005
batch size: 64
epoch:1, loss_train:0.3341, acc_train:0.8870, loss_valid:0.3632, acc_valid:0.8778, time_used:0.2820
batch size: 128
epoch:1, loss_train:0.3330, acc_train:0.8870, loss_valid:0.3630, acc_valid:0.8778, time_used:0.2089
batch size: 256
epoch:1, loss_train:0.3330, acc_train:0.8870, loss_valid:0.3629, acc_valid:0.8778, time_used:0.1774
batch size: 512
epoch:1, loss_train:0.3334, acc_train:0.8869, loss_valid:0.3629, acc_valid:0.8778, time_used:0.1693
batch size: 1024
epoch:1, loss_train:0.3336, acc_train:0.8869, loss_valid:0.3629, acc_valid:0.8778, time_used:0.1654
batch size: 2048
epoch:1, loss_train:0.3376, acc_train:0.8870, loss_valid:0.3629, acc_valid:0.8778, time_used:0.1696
batch size: 4096
epoch:1, loss_train:0.3335, acc_train:0.8870, loss_valid:0.3629, acc_valid:0.8778, time_used:0.1775
batch size: 8192
epoch:1, loss_train:0.3397, acc_train:0.8870, loss_valid:0.3628, acc_valid:0.8778, time_used:0.1862
batch size: 16384
epoch:1, loss_train:0.3337, acc_train:0.8870, loss_valid:0.3628, acc_valid:0.8778, time_used:0.1925
'''