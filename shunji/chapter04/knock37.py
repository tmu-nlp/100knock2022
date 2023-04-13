from knock30 import sentences
from collections import Counter
from matplotlib import pyplot as plt
import japanize_matplotlib

words = []
tmp = []
flg = False

for sentence in sentences:
    for d in sentence:
        if d['pos'] != '記号':
            if d['surface'] == '猫':
                flg = True
                continue
            tmp.append(d['surface'])
    if flg:
        words += tmp
    tmp = []
    flg = False    

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