import torch
from torch import nn
import pickle

model = pickle.load(open("SLP.pkl", "rb"))
X_train = torch.load("X_train.pt")
y_train = torch.load("y_train.pt")

criterion = nn.CrossEntropyLoss()

# l1のクロスエントロピー誤差
l1 = criterion(model(X_train[0]), y_train[0])
model.zero_grad()
l1.backward()
print("クロスエントロピー損失:", l1.item())
print("勾配", model.fc.weight.grad)

# l1からl4のクロスエントロピー誤差
l = criterion(model(X_train[:4]), y_train[:4])
model.zero_grad()
l.backward()
print("クロスエントロピー損失:", l.item())
print("勾配", model.fc.weight.grad)
