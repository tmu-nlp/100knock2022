import torch
import pandas as pd
from torch import nn
from collections import defaultdict
from knock80 import voc_id
from knock81 import RNN, CreateDataset, tokenizer
from knock82 import calculate_loss_and_accuracy, train_model, plot_result


class Padsequence():
    """バッチサイズでまとめるために系列長を揃える"""
    def __init__(self, padding_idx):
        self.padding_idx = padding_idx
    
    def __call__(self, batch):
        sorted_batch = sorted(batch, key=lambda x: x['inputs'].shape[0], reverse=True)
        sequences = [x['inputs'] for x in sorted_batch]
        sequence_padded = nn.utils.rnn.pad_sequence(sequences, batch_first=True, padding_value=self.padding_idx)  # padding（足りない分を0で補完）
        labels = torch.LongTensor([x['labels'] for x in sorted_batch])  # 浮動小数64bitのTensorを生成
        call = {}
        call['inputs'] = sequence_padded  # padding後のセンテンスに更新
        call['labels'] = labels  # labelsはそのまま
        return call

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

    # パラメータ設定
    VOCAB_SIZE = len(set(voc_id.values())) + 1  # 語彙サイズ
    EMB_SIZE = 300  # 埋め込み次元
    PADDING_IDX = len(set(voc_id.values()))
    OUTPUT_SIZE = 4  # 出力次元
    HIDDEN_SIZE = 50  # 隠れ層の次元
    NUM_EPOCHS = 10  # epoch数
    LERNING_RATE = 5e-2  # 学習率
    BATCH_SIZE = 32  # バッチサイズ


    # モデル
    model = RNN(HIDDEN_SIZE, VOCAB_SIZE, EMB_SIZE, PADDING_IDX, OUTPUT_SIZE)

    # 損失関数
    criterion = nn.CrossEntropyLoss()

    # オプティマイザ
    optimizer = torch.optim.SGD(model.parameters(), lr=LERNING_RATE)

    # 学習
    log_train, log_valid = train_model(dataset_train, dataset_valid, BATCH_SIZE, model, criterion, optimizer, NUM_EPOCHS, collate_fn=Padsequence(PADDING_IDX))

    plot_result(log_train, log_valid, "83.png")
    
    _, accuracy_train = calculate_loss_and_accuracy(model, dataset_train)
    _, accuracy_valid = calculate_loss_and_accuracy(model, dataset_valid)
    print(f'train_data accuracy : {accuracy_train:.4f}')
    print(f'valid_data accuracy : {accuracy_valid:.4f}')