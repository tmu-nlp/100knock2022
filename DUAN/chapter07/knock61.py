from gensim.models import KeyedVectors
m = KeyedVectors.load_word2vec_format('./100knock2022/DUAN/chapter07/GoogleNews-vectors-negative300.bin', binary=True)
print(m.similarity('United_States','U.S.'))