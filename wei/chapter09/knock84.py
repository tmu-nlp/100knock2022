'''
84. 単語ベクトルの導入
学習済みベクトルで単語埋め込みemb(x)を初期化し、学習を行う
'''
import pandas as pd
import numpy as np
import torch
from gensim.models import KeyedVectors
from knock80 import *    # made ids for words
from knock81 import *    # defined the RNN model and Dataset
#from knock82 import cal_loss_acc, train_model    # defined how to train the model




if __name__ == '__main__':
    file_path = '../data/GoogleNews-vectors-negative300.bin.gz'
    w2v_model = KeyedVectors.load_word2vec_format(file_path, binary=True)
    VOCAB_SIZE = len(set(word2id.values())) + 1
    EMB_SIZE = 300
    weights = np.zeros((VOCAB_SIZE, EMB_SIZE))
    cnt_words = 0
    for i, word in enumerate(word2id.keys()):
        try:
            weights[i] = w2v_model[word]  # 学習済みベクトルを付与
            cnt_words += 1
        except KeyError:
            weights[i] = np.random.normal(scale=0.4, size=(EMB_SIZE,))
    weights = torch.from_numpy(weights.astype(np.float32))  # change to tensor
    torch.save(weights, 'knock84_weights.pt')

    print(f'rate of learned words:{cnt_words / VOCAB_SIZE}')
    print(weights.size())


'''
rate of learned words:0.8983964248159831
torch.Size([7608, 300])
'''