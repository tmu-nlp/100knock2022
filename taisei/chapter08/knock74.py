import torch
import numpy as np
import joblib
from sklearn.metrics import accuracy_score

X_train = joblib.load("./output/X_train.joblib")
Y_train = joblib.load("./output/Y_train.joblib")
X_test = joblib.load("./output/X_test.joblib")
Y_test = joblib.load("./output/Y_test.joblib")

X_train = np.array(list(X_train.values))
Y_train = Y_train.values
X_test = np.array(list(X_test.values))
Y_test = Y_test.values

X_train_tenso = torch.from_numpy(X_train.astype(np.float32))
Y_train_tenso = torch.from_numpy(Y_train.astype(np.int64))
X_test_tenso = torch.from_numpy(X_test.astype(np.float32))
Y_test_tenso = torch.from_numpy(Y_test.astype(np.int64))

torch.manual_seed(0)
#モデルの読み込みにはあらかじめ、モデルと同じ形をしたインスタンスを用意。load_state_dictでパラメータの値を読み込む
net = torch.nn.Linear(300, 4)
net.load_state_dict(torch.load("./output/knock73_model.pth"))

axis = 1
#torch.maxは入力テンソルのすべての要素（行ごとか列ごとかは代2引数で指定？？）の最大値とそのインデックスを返す
Y_max_train, Y_pred_train = torch.max(net(X_train_tenso), axis)
print(f'学習データの正解率{accuracy_score(Y_pred_train, Y_train_tenso)}')

Y_max_test, Y_pred_test = torch.max(net(X_test_tenso), axis)
print(f'評価データの正解率{accuracy_score(Y_pred_test, Y_test_tenso)}')

"""
学習データの正解率0.7782046476761619
評価データの正解率0.7683658170914542
"""