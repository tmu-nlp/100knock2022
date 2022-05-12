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

print(fre)