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

noun_process = []
noun_ans = []
for i in range(len(ans)):
    if ans[i]["pos"] == "名詞":
        noun_process.append(ans[i]["surface"])
    elif len(noun_process) >= 2:
        noun_process = "".join(noun_process)
        noun_ans.append(noun_process)
        noun_process = []
    else:
        noun_process = []

for i in range(len(noun_ans)):
    print(noun_ans[i])