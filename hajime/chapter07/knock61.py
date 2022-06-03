from gensim.models import KeyedVectors
import pickle

with open("word2vec.pkl", "rb") as f:
    model = pickle.load(f)

sim = model.similarity("United_States", "U.S.")
print(sim)

"""
0.73107743
"""
