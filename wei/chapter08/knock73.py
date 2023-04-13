'''
73. 確率的勾配降下法(SDG)による学習
SDGを用いて、行列Wを学習(100エポックで終了)
'''
from torch.utils.data import TensorDataset,DataLoader
import torch
from knock71 import sglNN


X_train = torch.load('X_train.pt')
X_valid = torch.load('X_valid.pt')
y_train = torch.load('y_train.pt')
y_valid = torch.load('y_valid.pt')

# make dataset and dataloader(iterable data container)
my_data_train = TensorDataset(X_train, y_train)
d_loader_train = DataLoader(my_data_train, batch_size=1, shuffle=True, drop_last=False)  # epoch毎にshuffled,
my_data_valid = TensorDataset(X_valid, y_valid)
d_loader_valid = DataLoader(my_data_valid, batch_size=len(my_data_valid), shuffle=False, drop_last=False)

my_nn = sglNN(300, 4)
criterion = torch.nn.CrossEntropyLoss()
optimizer = torch.optim.SGD(my_nn.parameters(), lr=1e-1)

# training
num_epochs = 10
for epoch in range(1, num_epochs+1):
    # training mode
    my_nn.train()
    loss_train = 0.0
    for i, (inputs, labels) in enumerate(d_loader_train):
        optimizer.zero_grad()

        outputs = my_nn(inputs)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()

        loss_train += loss.item()
    loss_train = loss_train/i        # 全事例のロス総和/全事例数

    # validation mode
    my_nn.eval()
    with torch.no_grad():
        inputs, labels = next(iter(d_loader_valid))
        outputs = my_nn(inputs)
        loss_valid = criterion(outputs, labels)

    print(f'epoch:{epoch}, loss_train:{loss_train:.4f}, loss_valid:{loss_valid:.4f}')

torch.save(my_nn.state_dict(), 'knock73_model.bin')

'''
epoch:1, loss_train:0.5137, loss_valid:0.3967
epoch:2, loss_train:0.3636, loss_valid:0.3585
epoch:3, loss_train:0.3321, loss_valid:0.3432
epoch:4, loss_train:0.3151, loss_valid:0.3339
epoch:5, loss_train:0.3038, loss_valid:0.3298
epoch:6, loss_train:0.2961, loss_valid:0.3235
epoch:7, loss_train:0.2899, loss_valid:0.3257
epoch:8, loss_train:0.2855, loss_valid:0.3228
epoch:9, loss_train:0.2817, loss_valid:0.3210
epoch:10, loss_train:0.2778, loss_valid:0.3212
'''


