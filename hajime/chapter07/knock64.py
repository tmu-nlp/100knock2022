from gensim.models import KeyedVectors
import pickle

with open("word2vec.pkl", "rb") as f:
    model = pickle.load(f)

with open("64.txt", "w") as o_file:
    with open("questions-words.txt", "r") as t_file:
        for line in t_file:
            words = line.strip().split(" ")
            if len(words) != 4:
                continue
            key, value = model.most_similar(
                positive=[words[1], words[2]], negative=[words[0]])[0]
            o_file.write(
                f"{words[0]} {words[1]} {words[2]} {words[3]} {key} {value}\n")
