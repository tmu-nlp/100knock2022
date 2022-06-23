from knock73 import *


acc_train = calc_acc(model, dataloader_train)
acc_test = calc_acc(model, dataloader_test)

print(f"train_acc : {acc_train}")
print(f"test_acc : {acc_test}")

"""
train_acc : 0.9206289779108948
test_acc : 0.9034431137724551
"""
