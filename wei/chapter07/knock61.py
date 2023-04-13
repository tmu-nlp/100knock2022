'''
61. 単語の類似度
“United States”と”U.S.”のコサイン類似度を計算
'''


from knock60 import *


if __name__ == '__main__':

    wv = load_wv()
    sim = wv.similarity('United_States','U.S.')
    print(f'cosine similarity is {sim:.4f}')
    # cosine similarity is 0.7311

