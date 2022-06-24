'''
75.損失と正解率のプロット
各エポックのパラメータ更新が完了するたびに，訓練データでの損失，正解率，検証データでの損失，正解率をグラフにプロットし，
学習の進捗状況を確認できるようにせよ
'''
from knock71 import sglNN
import torch
from torch.utils.data import TensorDataset,DataLoader
from matplotlib import pyplot as plt

def cal_loss_acc(model, criterion, loader):
    model.eval()
    loss = float(0)
    total = 0
    correct = 0
    with torch.no_grad():
        for inputs, labels in loader:
            outputs = model(inputs)
            loss += criterion(outputs, labels).item()
            pred = torch.argmax(outputs, dim=-1)
            total += len(inputs)
            correct += (pred == labels).sum().item()
    return loss/len(loader), correct/total

if __name__ == '__main__':

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
    my_nn.load_state_dict(torch.load(model_path))   # load trained model ??

    criterion = torch.nn.CrossEntropyLoss()
    optimizer = torch.optim.SGD(my_nn.parameters(), lr=1e-1)

    num_epochs = 10
    train_log = []
    valid_log = []
    for epoch in range(1, num_epochs+1):
        # training mode
        my_nn.train()

        for i, (inputs, labels) in enumerate(d_loader_train):
            optimizer.zero_grad()

            outputs = my_nn(inputs)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()

        loss_train, acc_train = cal_loss_acc(my_nn, criterion, d_loader_train)
        loss_valid, acc_valid = cal_loss_acc(my_nn, criterion, d_loader_valid)
        train_log.append([loss_train, acc_train])
        valid_log.append([loss_valid, acc_valid])

        print(f'epoch:{epoch}, loss_train:{loss_train:.4f}, acc_train:{acc_train:.4f}, loss_valid:{loss_valid:.4f}, acc_valid:{acc_valid:.4f}')
'''
epoch:1, loss_train:0.2719, acc_train:0.9092, loss_valid:0.3325, acc_valid:0.8846
epoch:2, loss_train:0.2648, acc_train:0.9084, loss_valid:0.3251, acc_valid:0.8891
epoch:3, loss_train:0.2630, acc_train:0.9080, loss_valid:0.3250, acc_valid:0.8876
epoch:4, loss_train:0.2611, acc_train:0.9095, loss_valid:0.3258, acc_valid:0.8868
epoch:5, loss_train:0.2608, acc_train:0.9087, loss_valid:0.3263, acc_valid:0.8898
epoch:6, loss_train:0.2590, acc_train:0.9106, loss_valid:0.3267, acc_valid:0.8876
epoch:7, loss_train:0.2577, acc_train:0.9105, loss_valid:0.3249, acc_valid:0.8898
epoch:8, loss_train:0.2574, acc_train:0.9117, loss_valid:0.3264, acc_valid:0.8891
epoch:9, loss_train:0.2593, acc_train:0.9110, loss_valid:0.3349, acc_valid:0.8868
epoch:10, loss_train:0.2542, acc_train:0.9118, loss_valid:0.3270, acc_valid:0.8913
'''
