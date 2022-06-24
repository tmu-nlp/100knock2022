def calclate_acuracy(model, loader):
    model.eval()
    total, correct = 0, 0
    with torch.no_grad():
        for inputs, labels in loader:
            outputs = model(inputs)
            pred = torch.argmax(outputs, dim=-1)
            total += len(inputs)
            correct += (pred == labels).sum().item()
    accuracy = correct / total
    return accuracy

train_accuracy = calclate_acuracy(model, train_dataloader)
test_accuracy = calclate_acuracy(model, test_dataloader)

print(f'正解率(訓練データ): {train_accuracy}')
print(f'正解率(テストデータ): {test_accuracy}')

'''
正解率(訓練データ): 0.9323463268365817
正解率(テストデータ): 0.8995502248875562
'''
