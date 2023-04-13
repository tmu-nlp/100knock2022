from knock71 import NeuralNetwork
from knock73 import X_train, Y_train, X_valid, Y_valid, X_test, Y_test, train_dataloader, test_dataloader
import torch
from torch.utils.data import DataLoader

def calculate_accuracy(model, loader):
    model.eval()
    total = 0
    correct = 0
    with torch.no_grad():
        for inputs, labels in loader:
            outputs = model(inputs)
            pred = torch.argmax(outputs, dim=-1)
            total += len(inputs)
            correct += (pred == labels).sum().item()
    return correct / total

# モデル
model = torch.load("model.pt")

train_accuracy = calculate_accuracy(model, train_dataloader)
test_accuracy = calculate_accuracy(model, test_dataloader)

print(f'train_accuracy : {train_accuracy:.3f}')
print(f'test_accuracy : {test_accuracy:.3f}')

"""
train_accuracy : 0.889
test_accuracy : 0.858
"""