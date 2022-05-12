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

for i in range(len(ans)):
    if ans[i]["surface"] == "の" and ans[i - 1]["pos"] == "名詞" and ans[i + 1]["pos"] == "名詞":
        print(ans[i - 1]["surface"] + ans[i]["surface"] + ans[i + 1]["surface"])