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


def train_model(my_data_train, my_data_valid, batch_size, num_epochs, model, optimizer, criterion):
    # batch毎に、学習に使うデータ量が違うため、DataLoaderを作る時に決める
    d_loader_train = DataLoader(my_data_train, batch_size=batch_size, shuffle=True, drop_last=False)  # epoch毎にshuffled
    d_loader_valid = DataLoader(my_data_valid, batch_size=len(my_data_valid), shuffle=False, drop_last=False)
    for epoch in range(1, num_epochs+1):
        start = time.time()
        model.train()
        for inputs, labels in d_loader_train:
            optimizer.zero_grad()

            outputs = model.forward(inputs)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()

        loss_train, acc_train = cal_loss_acc(model, criterion, d_loader_train)
        loss_valid, acc_valid = cal_loss_acc(model, criterion, d_loader_valid)
        # save checkpoints
        # model_param_dic = {'epoch': epoch, 'model_state_dict': model.state_dict(), 'optimizer_state_dic': optimizer.state_dict()}
        # torch.save(model_param_dic, f'knock77_checkpoint_{batch_size}_{epoch}.pth')

        end = time.time()
        time_used = end - start

        print(f'epoch:{epoch}, loss_train:{loss_train:.4f}, acc_train:{acc_train:.4f}, loss_valid:{loss_valid:.4f}, acc_valid:{acc_valid:.4f}, time_used:{time_used:.3f}')

    #return {'train_log': log_train, 'valid_log': log_valid}


if __name__ == '__main__'    :
    X_train = torch.load('X_train.pt')
    X_valid = torch.load('X_valid.pt')
    y_train = torch.load('y_train.pt')
    y_valid = torch.load('y_valid.pt')
    my_data_train = TensorDataset(X_train, y_train)
    my_data_valid = TensorDataset(X_valid, y_valid)
    for batch_size in [2**i for i in range(11)]:
        my_nn = sglNN(300, 4)
        criterion = torch.nn.CrossEntropyLoss()
        optimizer = torch.optim.SGD(my_nn.parameters(), lr=1e-1)
        print(f'batch size: {batch_size}')
        train_model(my_data_train, my_data_valid, batch_size=batch_size, num_epochs=10, model=my_nn, criterion=criterion, optimizer=optimizer)
'''

'''