from gensim.models import KeyedVectors
m = KeyedVectors.load_word2vec_format('./100knock2022/DUAN/chapter07/GoogleNews-vectors-negative300.bin', binary=True)
print(m.most_similar('United_States', topn=10))