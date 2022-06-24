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


if __name__ == '__main__'    :
    X_train = torch.load('X_train.pt')
    X_valid = torch.load('X_valid.pt')
    y_train = torch.load('y_train.pt')
    y_valid = torch.load('y_valid.pt')
    my_data_train = TensorDataset(X_train, y_train)
    my_data_valid = TensorDataset(X_valid, y_valid)

    my_nn = sglNN(300, 4)
    criterion = torch.nn.CrossEntropyLoss()
    optimizer = torch.optim.SGD(my_nn.parameters(), lr=1e-1)

    for batch_size in [2**i for i in range(11)]:
        print(f'batch size: {batch_size}')
        train_model(my_data_train, my_data_valid, batch_size=batch_size, num_epochs=10, model=my_nn, criterion=criterion, optimizer=optimizer)
'''
batch size: 1
epoch:1, loss_train:0.3876, acc_train:0.8663, loss_valid:0.4312, acc_valid:0.8463, time_used:4.485
epoch:2, loss_train:0.3358, acc_train:0.8847, loss_valid:0.3814, acc_valid:0.8666, time_used:3.293
epoch:3, loss_train:0.3166, acc_train:0.8898, loss_valid:0.3612, acc_valid:0.8718, time_used:3.363
epoch:4, loss_train:0.3021, acc_train:0.8955, loss_valid:0.3510, acc_valid:0.8763, time_used:3.268
epoch:5, loss_train:0.2923, acc_train:0.8996, loss_valid:0.3403, acc_valid:0.8853, time_used:3.382
epoch:6, loss_train:0.2856, acc_train:0.9016, loss_valid:0.3341, acc_valid:0.8846, time_used:3.377
epoch:7, loss_train:0.2811, acc_train:0.9013, loss_valid:0.3316, acc_valid:0.8928, time_used:3.304
epoch:8, loss_train:0.2764, acc_train:0.9046, loss_valid:0.3342, acc_valid:0.8846, time_used:3.310
epoch:9, loss_train:0.2756, acc_train:0.9040, loss_valid:0.3327, acc_valid:0.8868, time_used:3.310
epoch:10, loss_train:0.2715, acc_train:0.9058, loss_valid:0.3322, acc_valid:0.8838, time_used:3.258
batch size: 32
epoch:1, loss_train:0.2541, acc_train:0.9112, loss_valid:0.3259, acc_valid:0.8921, time_used:0.301
epoch:2, loss_train:0.2539, acc_train:0.9115, loss_valid:0.3258, acc_valid:0.8921, time_used:0.327
epoch:3, loss_train:0.2542, acc_train:0.9115, loss_valid:0.3258, acc_valid:0.8921, time_used:0.332
epoch:4, loss_train:0.2540, acc_train:0.9116, loss_valid:0.3258, acc_valid:0.8928, time_used:0.334
epoch:5, loss_train:0.2542, acc_train:0.9116, loss_valid:0.3258, acc_valid:0.8921, time_used:0.340
epoch:6, loss_train:0.2538, acc_train:0.9114, loss_valid:0.3259, acc_valid:0.8921, time_used:0.303
epoch:7, loss_train:0.2539, acc_train:0.9116, loss_valid:0.3260, acc_valid:0.8921, time_used:0.328
epoch:8, loss_train:0.2538, acc_train:0.9117, loss_valid:0.3259, acc_valid:0.8921, time_used:0.401
epoch:9, loss_train:0.2540, acc_train:0.9120, loss_valid:0.3258, acc_valid:0.8928, time_used:0.384
epoch:10, loss_train:0.2541, acc_train:0.9117, loss_valid:0.3259, acc_valid:0.8928, time_used:0.299
batch size: 128
epoch:1, loss_train:0.2533, acc_train:0.9116, loss_valid:0.3259, acc_valid:0.8928, time_used:0.269
epoch:2, loss_train:0.2538, acc_train:0.9119, loss_valid:0.3259, acc_valid:0.8928, time_used:0.242
epoch:3, loss_train:0.2541, acc_train:0.9116, loss_valid:0.3259, acc_valid:0.8928, time_used:0.239
epoch:4, loss_train:0.2529, acc_train:0.9119, loss_valid:0.3259, acc_valid:0.8928, time_used:0.297
epoch:5, loss_train:0.2529, acc_train:0.9117, loss_valid:0.3260, acc_valid:0.8928, time_used:0.279
epoch:6, loss_train:0.2537, acc_train:0.9117, loss_valid:0.3260, acc_valid:0.8928, time_used:0.260
epoch:7, loss_train:0.2538, acc_train:0.9119, loss_valid:0.3260, acc_valid:0.8928, time_used:0.291
epoch:8, loss_train:0.2540, acc_train:0.9116, loss_valid:0.3260, acc_valid:0.8928, time_used:0.245
epoch:9, loss_train:0.2543, acc_train:0.9117, loss_valid:0.3260, acc_valid:0.8928, time_used:0.279
epoch:10, loss_train:0.2544, acc_train:0.9119, loss_valid:0.3260, acc_valid:0.8928, time_used:0.214
batch size: 1024
epoch:1, loss_train:0.2541, acc_train:0.9117, loss_valid:0.3260, acc_valid:0.8928, time_used:0.200
epoch:2, loss_train:0.2546, acc_train:0.9118, loss_valid:0.3260, acc_valid:0.8928, time_used:0.306
epoch:3, loss_train:0.2547, acc_train:0.9117, loss_valid:0.3260, acc_valid:0.8928, time_used:0.243
epoch:4, loss_train:0.2559, acc_train:0.9117, loss_valid:0.3260, acc_valid:0.8928, time_used:0.250
epoch:5, loss_train:0.2545, acc_train:0.9116, loss_valid:0.3260, acc_valid:0.8928, time_used:0.235
epoch:6, loss_train:0.2509, acc_train:0.9117, loss_valid:0.3260, acc_valid:0.8928, time_used:0.223
epoch:7, loss_train:0.2526, acc_train:0.9116, loss_valid:0.3260, acc_valid:0.8928, time_used:0.239
epoch:8, loss_train:0.2502, acc_train:0.9117, loss_valid:0.3260, acc_valid:0.8928, time_used:0.225
epoch:9, loss_train:0.2512, acc_train:0.9117, loss_valid:0.3260, acc_valid:0.8928, time_used:0.243
epoch:10, loss_train:0.2521, acc_train:0.9117, loss_valid:0.3260, acc_valid:0.8928, time_used:0.245
'''