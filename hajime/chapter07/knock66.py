import pickle
from scipy.stats import spearmanr

with open("word2vec.pkl", "rb") as f:
    model = pickle.load(f)

word1 = []
word2 = []
human_rank = []
model_rank = []

with open("combined.csv", "r") as f:
    next(f)
    for line in f:
        line_list = line.strip().split(",")
        word1.append(line_list[0])
        word2.append(line_list[1])
        human_rank.append(line_list[2])
        model_rank.append(model.similarity(line_list[0], line_list[1]))

correlation, pvalue = spearmanr(human_rank, model_rank)
print(correlation)

"""
0.6849564489532377
"""
