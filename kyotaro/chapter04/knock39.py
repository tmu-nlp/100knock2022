import matplotlib.pyplot as plt
import japanize_matplotlib
from collections import defaultdict

dic = dict([("surface", 0), ("base", 0), ("pos", 0), ("pos1", 0)])
ans = []

with open("neko.txt.mecab", "r") as text:
    for line in text:
        if line != 'EOS\n':
            line = line.replace('\t', ',').split(',')
            if line[0] != '\n':
                dic["surface"] = line[0]
                dic["base"] = line[7]
                dic["pos"] = line[1]
                dic["pos1"] = line[2]
                if line[0] != '':
                  ans.append(dic.copy())

fre = defaultdict(lambda: 0)

for data in ans:
    if data["pos"] != "記号":
        fre[data["surface"]] += 1

fre = sorted(fre.items(), key = lambda x: x[1], reverse = True)

key = [i for i in range(1, len(fre) + 1)]

value = []
for data in fre:
    value.append(data[1])


plt.figure(figsize = (8, 4))
plt.scatter(key, value)

ax = plt.gca()
ax.set_yscale('log')
ax.set_xscale('log')
plt.show()