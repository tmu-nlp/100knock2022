#参考記事
#https://note.nkmk.me/python-pytorch-module-sequential/s
#https://qiita.com/hs777/items/8238d1445de4c7472c21
import pickle
import pandas as pd
import numpy as np
import torch
import joblib

if __name__ == "__main__":
    X_train = joblib.load("./output/X_train.joblib")

    #X_trainをpandasのseries型からnumpyに変換。X_train.valuesだけだと中身の配列が
    X_train = np.array(list(X_train.values))

    #テンソルに変換(float64だとdoubleだからエラーが出る。だからfloat32に変換)
    X_train_tenso = torch.from_numpy(X_train.astype(np.float32)) 

    torch.manual_seed(0)
    net = torch.nn.Linear(300, 4)

    x1 = torch.reshape(X_train_tenso[0], (1, 300)) ##X_train_tenso[0]のshapeは300になってるから変換する

    #dimの次元で和を求めると確率の総和だから1になる
    Y_pred1 = torch.softmax(net.forward(x1), dim=-1)
    Y_pred4 = torch.softmax(net.forward(X_train_tenso[:4]), dim=-1)

    print(Y_pred1)
    print(Y_pred4)

"""
tensor([[0.2435, 0.2506, 0.2621, 0.2439]], grad_fn=<SoftmaxBackward0>)
tensor([[0.2435, 0.2506, 0.2621, 0.2439],
        [0.2362, 0.2655, 0.2565, 0.2418],
        [0.2563, 0.2532, 0.2506, 0.2398],
        [0.2423, 0.2634, 0.2551, 0.2392]], grad_fn=<SoftmaxBackward0>)
"""