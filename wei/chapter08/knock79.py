'''
79. multilayer NN
'''
import torch
import datetime
from torch.utils.data import TensorDataset, DataLoader
from torch.nn import functional as F
from knock75 import cal_loss_acc
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

    for batch_size in [2**i for i in range(5)]:
        # batch_size毎に、学習に使うデータ量が違うため、DataLoaderを作る時に決める
        my_data_train = TensorDataset(X_train, y_train)
        d_loader_train = DataLoader(my_data_train, batch_size=batch_size, shuffle=True, drop_last=False)  # epoch毎にshuffled
        my_data_valid = TensorDataset(X_valid, y_valid)
        d_loader_valid = DataLoader(my_data_valid, batch_size=len(my_data_valid), shuffle=False, drop_last=False)

        my_nn = MLPNet(300, 200, 1, 4)
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        criterion = torch.nn.CrossEntropyLoss()
        optimizer = torch.optim.SGD(my_nn.parameters(), lr=1e-3)

        print(f'batch size: {batch_size}')
        log = train_model(d_loader_train, d_loader_valid, batch_size=batch_size, num_epochs=1, model=my_nn, criterion=criterion, optimizer=optimizer, device=device)

