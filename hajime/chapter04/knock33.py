# knock-33
# 2つの名詞が「の」で連結されている名詞句を抽出せよ．

import knock30

for line in knock30.sentence_list:
    for i in range(len(line)-2):
        if line[i]["pos"] == "名詞" and line[i+1]["surface"] == "の" and line[i+2]["pos"] == "名詞":
            no = line[i]["surface"] + line[i+1]["surface"] + line[i+2]["surface"] 
            print(no)