import torch
from torch import nn
import pickle
from torch.utils.data import Dataset
from torch.utils.data import DataLoader


class NewsDataset(Dataset):
    def __init__(self, X, y):  # datasetの構成要素を指定
        self.X = X
        self.y = y

    def __len__(self):  # len(dataset)で返す値を指定
        return len(self.y)

    def __getitem__(self, idx):  # dataset[idx]で返す値を指定
        return [self.X[idx], self.y[idx]]


def picklize(obj_dic):
    """まとめてpickle化する関数"""
    for k, val in obj_dic.items():
        with open(str(k) + ".pkl", "wb") as f:
            pickle.dump(val, f)


if __name__ == "__main__":
    # 行列の読み込み
    X_train = torch.load("X_train.pt")
    X_valid = torch.load("X_valid.pt")
    X_test = torch.load("X_test.pt")
    y_train = torch.load("y_train.pt")
    y_valid = torch.load("y_valid.pt")
    y_test = torch.load("y_test.pt")

    # Datasetの作成
    dataset_train = NewsDataset(X_train, y_train)
    dataset_valid = NewsDataset(X_valid, y_valid)
    dataset_test = NewsDataset(X_test, y_test)

    # Dataloaderの作成
    dataloader_train = DataLoader(dataset_train, batch_size=10, shuffle=True)
    dataloader_valid = DataLoader(
        dataset_valid, batch_size=len(dataset_valid), shuffle=False
    )
    dataloader_test = DataLoader(
        dataset_test, batch_size=len(dataset_test), shuffle=False
    )

    # パーセプトロンの読み込み
    with open("SLP.pkl", "rb") as f:
        model = pickle.load(f)

    # 損失関数の定義
    criterion = nn.CrossEntropyLoss()

    # オプティマイザの定義
    optimizer = torch.optim.SGD(model.parameters(), lr=1e-1)

    # 学習
    epochs = 100
    for epoch in range(epochs):
        # 訓練モードに設定
        model.train()
        loss_train = 0.0
        for i, (inputs, labels) in enumerate(dataloader_train):
            # 勾配をゼロで初期化
            optimizer.zero_grad()

            # 順伝播 + 誤差逆伝播 + 重み更新
            outputs = model(inputs)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()

            # 損失を記録
            loss_train += loss.item()

        # バッチ単位の平均損失計算
        #   print(i)
        loss_train = loss_train / i

        # 検証データの損失計算
        model.eval()
        # 重み更新なしモード
        with torch.no_grad():
            inputs, labels = next(iter(dataloader_valid))
            outputs = model(inputs)
            loss_valid = criterion(outputs, labels)

        # ログを出力
        print(
            "epoch: {}, loss_train: {}, loss_valid: {}".format(
                epoch + 1, loss_train, loss_valid
            )
        )

    d = {
        "SLP_trained": model,
        "dataloader_train": dataloader_train,
        "dataloader_valid": dataloader_valid,
        "dataloader_test": dataloader_test,
        "dataset_train": dataset_train,
        "dataset_valid": dataset_valid,
        "dataset_test": dataset_test,
    }
    picklize(d)
