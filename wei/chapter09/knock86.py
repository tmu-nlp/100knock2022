""'''
86．畳込みニューラルネット
CNNを用いて、単語列ｘからカテゴリｙを予測するモデルを実装'''
from knock80 import *    # made ids for words
from knock81 import *    # defined the RNN model and Dataset
import torch
from torch.nn import functional as F
from gensim.models import KeyedVectors
import numpy as np

class CNN(torch.nn.Module):
    def __init__(self, emb_size, vocab_size, padding_idx, output_size, output_chanels, kernel_heights, stride, padding, emb_weights):
        super().__init__()
        if emb_weights != None:
            self.emb = torch.nn.Embedding.from_pretrained(
                emb_weights, padding_idx=padding_idx
            )
        else:
            self.emb = torch.nn.Embedding(vocab_size, emb_size, padding_idx=padding_idx)
        self.conv = torch.nn.Conv2d(1, output_chanels, (kernel_heights, emb_size), stride, (padding,0))
        self.drop = torch.nn.Dropout(0.3)    # 一部のneurons への　connectionを切断
        self.fc = torch.nn.Linear(output_chanels, output_size)

    def forward(self, x):
        emb = self.emb(x).unsqueeze(1)
        conv = self.conv(emb)
        act = F.relu(conv.squeeze(3))
        max_pool = F.max_pool1d(act, act.size()[2])
        output = self.fc(self.drop(max_pool.squeeze(2)))
        return output


if __name__ == '__main__':
    # set parameter
    VOCAB_SIZE = len(set(word2id.values())) + 1
    EMB_SIZE = 300
    PADDING_IDX = len(set(word2id.values()))
    OUTPUT_SIZE = 4
    OUTPUT_CHANNELS = 100
    KERNEL_HEIGHTS = 3    # filter_size
    STRIDE = 1
    PADDING = 1
    # BATCH_SIZE = 64
    # NUM_EPOCHS = 10
    # NUM_LAYERS = 2
    file_path = '../data/GoogleNews-vectors-negative300.bin.gz'
    w2v_model = KeyedVectors.load_word2vec_format(file_path, binary=True)
    weights = np.zeros((VOCAB_SIZE, EMB_SIZE))
    cnt_words = 0
    for i, word in enumerate(word2id.keys()):
        try:
            weights[i] = w2v_model[word]  # 学習済みベクトルを付与
            cnt_words += 1
        except KeyError:
            weights[i] = np.random.normal(scale=0.4, size=(EMB_SIZE,))
    WEIGHTS = torch.from_numpy(weights.astype(np.float32))  # change to tensor



    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    cnn = CNN(
        emb_size=EMB_SIZE, vocab_size=VOCAB_SIZE, padding_idx=PADDING_IDX, output_size=OUTPUT_SIZE,
        output_chanels=OUTPUT_CHANNELS, kernel_heights=KERNEL_HEIGHTS, stride=STRIDE, padding=PADDING, emb_weights= WEIGHTS
        )

    for i in range(10):
        x = dataset_train[i]['inputs']
        print(torch.softmax(cnn(x.unsqueeze(0)), dim=-1))

'''
tensor([[0.2472, 0.2209, 0.2827, 0.2492]], grad_fn=<SoftmaxBackward>)
tensor([[0.2651, 0.2284, 0.3049, 0.2016]], grad_fn=<SoftmaxBackward>)
tensor([[0.2624, 0.2361, 0.2958, 0.2056]], grad_fn=<SoftmaxBackward>)
tensor([[0.2063, 0.2433, 0.3044, 0.2459]], grad_fn=<SoftmaxBackward>)
tensor([[0.2474, 0.1984, 0.2806, 0.2736]], grad_fn=<SoftmaxBackward>)
tensor([[0.2555, 0.1965, 0.3004, 0.2475]], grad_fn=<SoftmaxBackward>)
tensor([[0.2871, 0.2140, 0.2892, 0.2097]], grad_fn=<SoftmaxBackward>)
tensor([[0.2586, 0.2106, 0.2611, 0.2696]], grad_fn=<SoftmaxBackward>)
tensor([[0.2663, 0.2110, 0.2869, 0.2359]], grad_fn=<SoftmaxBackward>)
tensor([[0.2988, 0.1804, 0.2863, 0.2346]], grad_fn=<SoftmaxBackward>)'''