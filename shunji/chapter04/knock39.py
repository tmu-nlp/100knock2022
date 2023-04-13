from knock30 import sentences
from collections import Counter
from matplotlib import pyplot as plt
import japanize_matplotlib

words = []

for sentence in sentences:
    for d in sentence:
        if d['pos'] != '記号':
            words.append(d['surface'])

c = Counter(words) # Counterオブジェクトの生成(dictのサブクラスで {要素: 出現回数, ...})
c = c.most_common() # 出現回数順の[(要素, 出現回数), ...] 

x = [i for i in range(1, len(c) + 1)]
y = []
for t in c:
    y.append(t[1])

fig, ax = plt.subplots()
ax.scatter(x, y)
ax.set_xscale('log')
ax.set_yscale('log')
plt.savefig('39.png')
plt.show()