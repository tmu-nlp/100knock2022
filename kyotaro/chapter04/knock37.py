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

cat = defaultdict(lambda: 0)

for i in range(len(ans)):
    if ans[i]["surface"] == "猫":
        if ans[i - 1]["pos"] != "記号":
            cat[ans[i - 1]["surface"]] += 1
        if ans[i + 1]["pos"] != "記号":
            cat[ans[i + 1]["surface"]] += 1

cat = sorted(cat.items(), key = lambda x: x[1], reverse = True)

key = [cat[i][0] for i in range(10)]
value = [cat[i][1] for i in range(10)]

plt.figure(figsize = (9, 4))
plt.bar(key, value)
plt.show()