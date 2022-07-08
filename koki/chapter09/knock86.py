from torch.nn import functional


class CNN(nn.Module):
    def __init__(self, vocab_size, emb_size, padding_idx, output_size,
                 out_channels, kernel_heights, stride, padding, emb_weights=None):
        super().__init__()
        if emb_weights != None:  # 指定があれば埋め込み層の重みをemb_weightsで初期化
            self.emb = nn.Embedding.from_pretrained(
                emb_weights, padding_idx=padding_idx)
        else:
            self.emb = nn.Embedding(
                vocab_size, emb_size, padding_idx=padding_idx)
        self.conv = nn.Conv2d(
            1, out_channels, (kernel_heights, emb_size), stride, (padding, 0))
        self.drop = nn.Dropout(0.3)
        self.fc = nn.Linear(out_channels, output_size)

    def forward(self, x):
        emb = self.emb(x).unsqueeze(1)
        conv = self.conv(emb)
        act = functional.relu(conv.squeeze(3))
        max_pool = functional.max_pool1d(act, act.size()[2])
        out = self.fc(self.drop(max_pool.squeeze(2)))
        return out


# パラメータの設定
VOCAB_SIZE = len(set(word_id.values())) + 1
EMB_SIZE = 300
PADDING_IDX = len(set(word_id.values()))
OUTPUT_SIZE = 4
OUT_CHANNELS = 100
KERNEL_HEIGHTS = 3
STRIDE = 1
PADDING = 1

# モデルの定義
model = CNN(VOCAB_SIZE, EMB_SIZE, PADDING_IDX, OUTPUT_SIZE,
            OUT_CHANNELS, KERNEL_HEIGHTS, STRIDE, PADDING, emb_weights=weights)

# 先頭10件の予測値取得
for i in range(10):
    X = dataset_train[i]['inputs']
    # unsqueezwはinputしたTensorの次元を1増やして返す
    print(torch.softmax(model(X.unsqueeze(0)), dim=-1))
