'''
79. multilayer NN
'''
import torch
from torch.utils.data import TensorDataset, DataLoader
from torch.nn import functional as F
from knock78 import train_model


# define multilayer NN
class MLPNet(torch.nn.Module):
    def __init__(self, input_size, hidden_size, hidden_layers, output_size):
        super().__init__()
        self.hidden_layers = hidden_layers
        self.fc = torch.nn.Linear(input_size, hidden_size)
        self.fc_hidden = torch.nn.Linear(hidden_size, hidden_size)
        self.fc_out = torch.nn.Linear(hidden_size, output_size)
        self.bn = torch.nn.BatchNorm1d(hidden_size)

    def forward(self, x):
        x = F.relu(self.fc(x))
        for _ in range(self.hidden_layers):
            x = F.relu(self.bn(self.fc_hidden(x)))
        x = F.relu(self.fc_out(x))

        return x

if __name__ == '__main__'    :
    X_train = torch.load('X_train.pt', map_location=lambda storage, loc: storage.cuda())
    X_valid = torch.load('X_valid.pt',map_location=lambda storage, loc: storage.cuda())
    y_train = torch.load('y_train.pt', map_location=lambda storage, loc: storage.cuda())
    y_valid = torch.load('y_valid.pt', map_location=lambda storage, loc: storage.cuda())

    my_nn = MLPNet(300, 200, 1, 4)
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    criterion = torch.nn.CrossEntropyLoss()
    optimizer = torch.optim.SGD(my_nn.parameters(), lr=1e-3)

    for batch_size in [2**i for i in range(6, 11)]:
        # batch_size毎に、学習に使うデータ量が違うため、DataLoaderを作る時に決める
        my_data_train = TensorDataset(X_train, y_train)
        d_loader_train = DataLoader(my_data_train, batch_size=batch_size, shuffle=True, drop_last=False)  # epoch毎にshuffled
        my_data_valid = TensorDataset(X_valid, y_valid)
        d_loader_valid = DataLoader(my_data_valid, batch_size=len(my_data_valid), shuffle=False, drop_last=False)
        print(f'batch size: {batch_size}')
        train_model(d_loader_train, d_loader_valid, batch_size=batch_size, num_epochs=10, model=my_nn, criterion=criterion, optimizer=optimizer, device=device)

'''
batch size: 64
epoch:1, loss_train:0.9520, acc_train:0.7210, loss_valid:0.9620, acc_valid:0.7189, time_used:0.6653
epoch:2, loss_train:0.7610, acc_train:0.7545, loss_valid:0.7726, acc_valid:0.7549, time_used:0.5329
epoch:3, loss_train:0.6812, acc_train:0.7656, loss_valid:0.6949, acc_valid:0.7609, time_used:0.4566
epoch:4, loss_train:0.6338, acc_train:0.7769, loss_valid:0.6492, acc_valid:0.7676, time_used:0.4360
epoch:5, loss_train:0.6006, acc_train:0.7879, loss_valid:0.6175, acc_valid:0.7826, time_used:0.4156
epoch:6, loss_train:0.5709, acc_train:0.7978, loss_valid:0.5910, acc_valid:0.7931, time_used:0.4229
epoch:7, loss_train:0.5493, acc_train:0.8046, loss_valid:0.5722, acc_valid:0.7969, time_used:0.4171
epoch:8, loss_train:0.5320, acc_train:0.8117, loss_valid:0.5571, acc_valid:0.7969, time_used:0.4139
epoch:9, loss_train:0.5147, acc_train:0.8163, loss_valid:0.5413, acc_valid:0.8028, time_used:0.4135
epoch:10, loss_train:0.4990, acc_train:0.8216, loss_valid:0.5273, acc_valid:0.8043, time_used:0.4264
batch size: 512
epoch:1, loss_train:0.3877, acc_train:0.8771, loss_valid:0.4225, acc_valid:0.8546, time_used:0.1961
epoch:2, loss_train:0.3867, acc_train:0.8768, loss_valid:0.4223, acc_valid:0.8546, time_used:0.1990
epoch:3, loss_train:0.3858, acc_train:0.8774, loss_valid:0.4207, acc_valid:0.8546, time_used:0.2243
epoch:4, loss_train:0.3839, acc_train:0.8773, loss_valid:0.4195, acc_valid:0.8546, time_used:0.2029
epoch:5, loss_train:0.3835, acc_train:0.8782, loss_valid:0.4188, acc_valid:0.8553, time_used:0.2179
epoch:6, loss_train:0.3824, acc_train:0.8785, loss_valid:0.4179, acc_valid:0.8561, time_used:0.2042
epoch:7, loss_train:0.3803, acc_train:0.8790, loss_valid:0.4164, acc_valid:0.8568, time_used:0.2188
epoch:8, loss_train:0.3798, acc_train:0.8797, loss_valid:0.4157, acc_valid:0.8576, time_used:0.2101
epoch:9, loss_train:0.3789, acc_train:0.8800, loss_valid:0.4146, acc_valid:0.8568, time_used:0.2053
epoch:10, loss_train:0.3777, acc_train:0.8802, loss_valid:0.4132, acc_valid:0.8583, time_used:0.2220
batch size: 1024
epoch:1, loss_train:0.3762, acc_train:0.8801, loss_valid:0.4132, acc_valid:0.8576, time_used:0.2328
epoch:2, loss_train:0.3766, acc_train:0.8800, loss_valid:0.4125, acc_valid:0.8576, time_used:0.2075
epoch:3, loss_train:0.3776, acc_train:0.8810, loss_valid:0.4122, acc_valid:0.8583, time_used:0.2291
epoch:4, loss_train:0.3752, acc_train:0.8811, loss_valid:0.4118, acc_valid:0.8583, time_used:0.2277
epoch:5, loss_train:0.3743, acc_train:0.8805, loss_valid:0.4109, acc_valid:0.8583, time_used:0.2286
epoch:6, loss_train:0.3734, acc_train:0.8817, loss_valid:0.4106, acc_valid:0.8583, time_used:0.2251
epoch:7, loss_train:0.3742, acc_train:0.8823, loss_valid:0.4103, acc_valid:0.8583, time_used:0.2088
epoch:8, loss_train:0.3753, acc_train:0.8817, loss_valid:0.4096, acc_valid:0.8583, time_used:0.2168
epoch:9, loss_train:0.3713, acc_train:0.8820, loss_valid:0.4090, acc_valid:0.8583, time_used:0.2283
epoch:10, loss_train:0.3714, acc_train:0.8824, loss_valid:0.4083, acc_valid:0.8591, time_used:0.2196
'''