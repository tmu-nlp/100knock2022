'''
60. 単語ベクトルの読み込みと表示
documents: https://radimrehurek.com/gensim/models/keyedvectors.html
store and query word vectors
classmethod load_word2vec_format(): load KeyedVectors from a file produces by the original C word2vec-tool format
'''

def load_wv():
    from gensim.models import keyedvectors

    file_path = '../data/GoogleNews-vectors-negative300.bin'

    wv_from_bin = keyedvectors.load_word2vec_format(file_path, binary=True)

    return wv_from_bin



if __name__ == '__main__':
    wv = load_wv()

    wv_us = wv['United_States']

    # print(wv_us.shape)
    # shape(300,) : numpy vector of a word


