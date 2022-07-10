import sys
import matplotlib.pyplot as plt
import re

"""
各バッチサイズのBLEUは以下のように表示されている
BLEU4 = 3.74, 12.6/5.4/2.5/1.2 (BP=1.000, ratio=2.479, syslen=54693, reflen=22063)
"""

path = sys.argv[1]
scores = []
pattern = 'BLEU4'

with open(path) as f:
    for line in f:
        if re.match(pattern, line):
            line = line.strip().split()
            score = re.sub(r',', '', line[2])
            score = float(score)
            scores.append(score)

labels = range(1, len(scores)+1)

fig = plt.figure()
plt.plot(labels, scores)
plt.xlabel('batch')
plt.ylabel('BLEU')
plt.show()
fig.savefig("./output/knock94_img.png")