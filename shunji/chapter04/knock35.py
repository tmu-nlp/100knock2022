from knock30 import sentences
from collections import Counter

words = []

for sentence in sentences:
    for d in sentence:
        if d['pos'] != '記号':
            words.append(d['surface'])

c = Counter(words)
print(*c.most_common(), sep='\n')