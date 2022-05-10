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
y = []
for i in range(10):
    x.append(c[i][0])
    y.append(c[i][1])

fig, ax = plt.subplots()
ax.bar(x, y)
plt.show()