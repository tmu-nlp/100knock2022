import torch
import pandas as pd
import numpy as np
from torch import nn
from collections import defaultdict
from gensim.models import KeyedVectors
from torch.nn import functional as F
from knock81 import CreateDataset, tokenizer
from knock82 import calculate_loss_and_accuracy, train_model, plot_result
from knock83 import Padsequence

class CNN(nn.Module):
    def __init__(self, vocab_size, emb_size, padding_idx, output_size, out_channels, kernel_heights, stride, padding, emb_weights=None):
        super().__init__()
        if emb_weights != None:
            self.emb = nn.Embedding.from_pretrained(emb_weights, padding_idx=padding_idx)
        else:
            self.emb = nn.Embedding(vocab_size, emb_size, padding_idx=padding_idx)
        self.conv = nn.Conv2d(1, out_channels, (kernel_heights, emb_size), stride, (padding, 0))
        self.drop = nn.Dropout(0.3)
        self.fc = nn.Linear(out_channels, output_size)
    
    def forward(self, x):
        emb = self.emb(x).unsqueeze(1)
        conv = self.conv(emb)
        act = F.relu(conv.squeeze(3))
        max_pool = F.max_pool1d(act, act.size()[2])
        out = self.fc(self.drop(max_pool.squeeze(2)))
        return out

if __name__ == "__main__":
    train = pd.read_csv("../chapter06/train.txt", sep="\t")
    valid = pd.read_csv("../chapter06/valid.txt", sep="\t")
    test = pd.read_csv("../chapter06/test.txt", sep="\t")

    frec = defaultdict(lambda: 0)  # 単語の頻度
    voc_id = defaultdict(lambda: 0)  # 単語のID

    for line in train["TITLE"]:
        words = line.strip().split()
        for word in words:
            frec[word] += 1

    frec = sorted(frec.items(), key=lambda x: x[1], reverse=True)  # 頻度順にソート

    for i, word in enumerate(frec):
        if word[1] >= 2:
            voc_id[word[0]] = i + 1


    # データをchapter06から持ってきて、ラベルを数値に変換
    category_dict = {'b': 0, 't': 1, 'e': 2, 'm': 3}

    y_train = pd.read_csv("../chapter06/train.txt", sep="\t")["CATEGORY"].map(lambda x: category_dict[x]).values
    y_valid = pd.read_csv("../chapter06/valid.txt", sep="\t")["CATEGORY"].map(lambda x: category_dict[x]).values
    y_test = pd.read_csv("../chapter06/test.txt", sep="\t")["CATEGORY"].map(lambda x: category_dict[x]).values

    # Datasetを作成
    dataset_train = CreateDataset(train["TITLE"], y_train, tokenizer, voc_id)
    dataset_valid = CreateDataset(valid["TITLE"], y_valid, tokenizer, voc_id)
    dataset_test = CreateDataset(test["TITLE"], y_test, tokenizer, voc_id)

    # 学習済み単語ベクトルをロード
    model = KeyedVectors.load_word2vec_format('GoogleNews-vectors-negative300.bin.gz', binary=True)

    # 学習済み単語ベクトルを取得
    VOCAB_SIZE = len(set(voc_id.values())) + 1
    EMB_SIZE = 300
    weights = np.zeros((VOCAB_SIZE, EMB_SIZE))  # 単語ベクトルを0で初期化
    words_in_pretrained = 0
    for i, word in enumerate(voc_id.keys()):
        try:
            weights[i] = model[word]  # i番目の単語の重みを更新
            words_in_pretrained += 1
        except KeyError:
            weights[i] = np.random.normal(scale=0.4, size=(EMB_SIZE,))  # もしないならランダムな値に更新
    weights = torch.from_numpy(weights.astype((np.float32)))  # numpy.ndarrayからTensorを生成

    # パラメータ設定
    VOCAB_SIZE = len(set(voc_id.values())) + 1  # 語彙サイズ
    EMB_SIZE = 300  # 埋め込み次元
    PADDING_IDX = len(set(voc_id.values()))
    OUTPUT_SIZE = 4  # 出力次元
    HIDDEN_SIZE = 100  # 隠れ層の次元
    NUM_EPOCHS = 10  # epoch数
    LERNING_RATE = 5e-2  # 学習率
    BATCH_SIZE = 32  # バッチサイズ
    NUM_LAYERS = 2  # 埋め込み層の数
    OUT_CHANNELS = 100
    KERNEL_HEIGHTS = 3
    STRIDE = 1
    PADDING = 1

    # モデルの定義
    model = CNN(VOCAB_SIZE, EMB_SIZE, PADDING_IDX, OUTPUT_SIZE, OUT_CHANNELS, KERNEL_HEIGHTS, STRIDE, PADDING, emb_weights=weights)

    # # 予測
    # for i in range(10):
    #     X = dataset_train[i]['inputs']
    #     print(torch.softmax(model(X.unsqueeze(0)), dim=-1))