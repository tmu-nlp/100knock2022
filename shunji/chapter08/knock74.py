"""
問題73で求めた行列を用いて学習データおよび評価データの事例を分類したとき，その正解率をそれぞれ求めよ．
"""
import torch
import pickle
from knock73 import NewsDataset  # unpickleの時に無いとダメらしい


def unpickle(fname_list):
    """一括で非pickle化する関数"""
    obj_list = []
    for fname in fname_list:
        with open(fname, "rb") as f:
            obj_list.append(pickle.load(f))
    return obj_list


def calculate_accuracy(model, loader):
    """modelとdataloaderを受け取って正解率を返す関数"""
    model.eval()  # evaluation mode
    total = 0
    correct = 0
    with torch.no_grad():  # 勾配計算しないブロック
        for inputs, labels in loader:
            outputs = model(inputs)
            pred = torch.argmax(outputs, dim=1)
            total += len(inputs)
            correct += (pred == labels).sum().item()  # 等価演算でTFのtensorが返され，Trueの数を加算

    return correct / total


model, dataloader_train, dataloader_test = unpickle(
    ["SLP_trained.pkl", "dataloader_train.pkl", "dataloader_test.pkl"]
)

acc_train = calculate_accuracy(model, dataloader_train)
acc_test = calculate_accuracy(model, dataloader_test)
print("accuracy(train): {}".format(acc_train))
print("accuracy(test): {}".format(acc_test))


"""
accuracy(train): 0.920727136431784
accuracy(test): 0.9017991004497751
"""
