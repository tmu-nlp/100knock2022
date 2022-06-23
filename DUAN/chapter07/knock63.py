from gensim.models import KeyedVectors
m = KeyedVectors.load_word2vec_format('./100knock2022/DUAN/chapter07/GoogleNews-vectors-negative300.bin', binary=True)
vec = m['Spain'] - m['madrid'] + m['Athens'] 
print(m.most_similar(positive=['Spain', 'Athens'], negative=['Madrid'], topn=10))