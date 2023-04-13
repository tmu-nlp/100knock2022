from knock30 import sentences
from collections import Counter
from matplotlib import pyplot as plt
import japanize_matplotlib

words = []

for sentence in sentences:
    for d in sentence:
        if d['pos'] != '記号':
            words.append(d['surface'])

c = Counter(words)
c = c.most_common()

x = []

for t in c:
    x.append(t[1])

fig, ax = plt.subplots()
ax.hist(x, bins=100)
plt.show()