from gensim.models import KeyedVectors
import pickle

model = KeyedVectors.load_word2vec_format(
    'GoogleNews-vectors-negative300.bin.gz', binary=True)

with open('GoogleNews-vectors.pkl', 'wb') as f:
    pickle.dump(model, f)
