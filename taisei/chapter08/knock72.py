import torch
import numpy as np
import joblib

if __name__ == "__main__":
    # データを読み込む
    X_train = joblib.load("./output/X_train.joblib")
    Y_train = joblib.load("./output/Y_train.joblib")

    X_train = np.array(list(X_train.values))
    Y_train = Y_train.values

    X_train_tenso = torch.from_numpy(X_train.astype(np.float32))
    Y_train_tenso = torch.from_numpy(Y_train.astype(np.int64))


    torch.manual_seed(0)
    net = torch.nn.Linear(300, 4)

    x1 = torch.reshape(X_train_tenso[0], (1, 300))

    Y_pred1 = torch.softmax(net.forward(x1), dim=-1)
    Y_pred4 = torch.softmax(net.forward(X_train_tenso[:4]), dim=-1)

    #平均損失を計算するためのもの（引数：モデルの出力、正解）
    loss_f = torch.nn.CrossEntropyLoss()

    #クロスエントロピー損失を求める
    loss_1 = loss_f(Y_pred1, Y_train_tenso[:1])
    loss_4 = loss_f(Y_pred4, Y_train_tenso[:4])

    #事例x1
    #テンソルに蓄積している勾配のバッファをすべて0に初期化
    net.zero_grad()
    #勾配を誤差逆伝播させる
    loss_1.backward()
    print(f"事例x1のクロスエントロピー損失：{loss_1}")
    print(f"事例x1の行列Wに対する勾配：{net.weight.grad}")

    #事例x1~x4
    net.zero_grad()
    loss_4.backward()
    print(f"事例x1~x4のクロスエントロピー損失：{loss_4}")
    print(f"事例x1~x4の行列Wに対する勾配：{net.weight.grad}")

"""
事例x1のクロスエントロピー損失：1.3928086757659912
事例x1の行列Wに対する勾配：tensor([[ 3.1695e-02,  1.5229e-03,  1.3748e-02,  ..., -3.2614e-03,
          2.5113e-04,  1.6504e-02],
        [-1.0479e-02, -5.0348e-04, -4.5451e-03,  ...,  1.0783e-03,
         -8.3026e-05, -5.4563e-03],
        [-1.1089e-02, -5.3278e-04, -4.8097e-03,  ...,  1.1410e-03,
         -8.7858e-05, -5.7738e-03],
        [-1.0128e-02, -4.8661e-04, -4.3929e-03,  ...,  1.0421e-03,
         -8.0244e-05, -5.2735e-03]])
事例x1~x4のクロスエントロピー損失：1.3931796550750732
事例x1~x4の行列Wに対する勾配：tensor([[ 1.4465e-02, -7.3031e-04, -6.2563e-05,  ..., -4.0307e-03,
         -3.5392e-03,  5.1752e-03],
        [-5.0492e-03,  4.2590e-05,  1.4307e-03,  ...,  1.5261e-03,
          2.9145e-03, -2.1139e-03],
        [-4.7249e-03,  6.6995e-04, -2.5928e-03,  ...,  1.1236e-03,
         -2.0167e-03, -1.0652e-03],
        [-4.6905e-03,  1.7770e-05,  1.2247e-03,  ...,  1.3810e-03,
          2.6415e-03, -1.9961e-03]])
"""