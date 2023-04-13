import pandas as pd
from collections import defaultdict
import re

def toknize(text, voc_id, unk=0):
    ids = []
    for word in text.split():
        ids.append(voc_id.get(word, unk))
    return ids


train = pd.read_csv("../chapter06/train.txt", sep="\t")

frec = defaultdict(lambda: 0)  # 単語の頻度
voc_id = defaultdict(lambda: 0)  # 単語のID

for line in train["TITLE"]:
    words = line.strip().split()
    for word in words:
        frec[word] += 1

frec = sorted(frec.items(), key=lambda x: x[1], reverse=True)  # 頻度順にソート

for i, word in enumerate(frec):
    if word[1] >= 2:
        voc_id[word[0]] = i + 1

text = train.iloc[3, train.columns.get_loc('TITLE')]
# print(f'text : {text}')
# print(toknize(text, voc_id))