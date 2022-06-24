from knock71 import NeuralNetwork
from knock73 import X_train, Y_train, X_valid, Y_valid, X_test, Y_test, train_dataloader, test_dataloader
import torch
from torch.utils.data import DataLoader

def calcurate_accuracy(model, loader):
    # modelの振る舞いを評価用に
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
model = NeuralNetwork(300, 4)

train_accuracy = calcurate_accuracy(model, train_dataloader)
test_accuracy = calcurate_accuracy(model, test_dataloader)

print(f'train_accuracy : {train_accuracy:.3f}')
print(f'test_accuracy : {test_accuracy:.3f}')


"""
train_accuracy : 0.3291791604197901
test_accuracy : 0.3073463268365817
"""