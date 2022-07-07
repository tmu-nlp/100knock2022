from knock72 import *
from torch.utils.data import Dataset
from torch.utils.data import DataLoader


dataset_train = NewsDataset(X_train, y_train)
dataset_valid = NewsDataset(X_valid, y_valid)
dataset_test = NewsDataset(X_test, y_test)

dataloader_train = DataLoader(dataset_train, batch_size=1, shuffle=True)
dataloader_valid = DataLoader(
    dataset_valid, batch_size=len(dataset_valid), shuffle=False)
dataloader_test = DataLoader(
    dataset_test, batch_size=len(dataset_test), shuffle=False)

# ここから学習
model = SingleLayerPerceptronNetwork(300, 4)
criterion = nn.CrossEntropyLoss()
optimizer = torch.optim.SGD(model.parameters(), lr=1e-1)
num_epochs = 10
for epoch in range(num_epochs):
    model.train()
    loss_train = 0.0
    for i, (inputs, labels) in enumerate(dataloader_train):
        optimizer.zero_grad()

        outputs = model(inputs)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()

        loss_train += loss.item()
    loss_train = loss_train / i
    model.eval()
    with torch.no_grad():
        inputs, labels = next(iter(dataloader_valid))
        outputs = model(inputs)
        loss_valid = criterion(outputs, labels)

    print(
        f'epoch: {epoch + 1}, loss_train: {loss_train:.4f}, loss_valid: {loss_valid:.4f}')

"""
epoch: 1, loss_train: 0.4772, loss_valid: 0.3298
epoch: 2, loss_train: 0.3146, loss_valid: 0.2987
epoch: 3, loss_train: 0.2866, loss_valid: 0.2880
epoch: 4, loss_train: 0.2708, loss_valid: 0.2758
epoch: 5, loss_train: 0.2605, loss_valid: 0.2791
epoch: 6, loss_train: 0.2543, loss_valid: 0.2715
epoch: 7, loss_train: 0.2485, loss_valid: 0.2716
epoch: 8, loss_train: 0.2448, loss_valid: 0.2710
epoch: 9, loss_train: 0.2420, loss_valid: 0.2733
epoch: 10, loss_train: 0.2386, loss_valid: 0.2707
"""
