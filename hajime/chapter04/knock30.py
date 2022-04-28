# knock-30
# 形態素解析結果（neko.txt.mecab）を読み込むプログラムを実装せよ．
# ただし，各形態素は表層形（surface），基本形（base），品詞（pos），品詞細分類1（pos1）をキーとするマッピング型に格納し，1文を形態素（マッピング型）のリストとして表現せよ．
# 第4章の残りの問題では，ここで作ったプログラムを活用せよ．
 
import re
import numpy as np

neko_file = open("neko.txt.mecab",'r')
with open('neko.txt.mecab','r') as f:
    neko_data = f.read()
split_neko = neko_data.split("\n")

# mecabの出力
# 表層形\t品詞,品詞細分類1,品詞細分類2,品詞細分類3,活用型,活用形,原形,読み,発音
# 一行を「/t」と「,」で分割
# dict[表層系，基本形，品詞，品詞細分類1] = 形態素（表層系） となるように計算
# EOSが読み込まれ次第新たなlistの要素にする

sentence_list = list()
dict_line = dict()
sentence = list()

for line in split_neko:
    split_line = re.split('[\t,]',line)
    if len(split_line) == 1 and split_line[0] == "":
        continue
    elif len(split_line) == 1 and split_line[0] == "EOS":
        sentence_list.append(sentence)
        sentence = list()
        continue
    dict_line["surface"] = split_line[0]
    dict_line["base"] = split_line[7]
    dict_line["pos"] = split_line[1]
    dict_line["pos1"] = split_line[2]
    sentence.append(dict_line)
    dict_line = dict()

# for line in sentence_list:
#     print(line)