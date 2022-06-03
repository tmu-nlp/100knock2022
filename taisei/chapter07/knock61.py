from gensim.models import KeyedVectors
import pickle

if __name__ == "__main__":
    with open("./output/knock60_word2vec", "rb") as f:
        model = pickle.load(f)

    cos_sim = model.similarity("United_States", "U.S.")
    print(cos_sim)

"""
0.73107743
"""