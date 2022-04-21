#!/usr/bin/python3

# regular expression の参考サイト
# https://note.nkmk.me/python-re-match-search-findall-etc/

import json
import gzip
import re

with gzip.open('jawiki-country.json.gz', 'rt') as f:
    for line in f:
        content = json.loads(line) # type-str so s is added
        if ("title","イギリス") in content.items():
            eng = content["text"]

# using regular expression
eng_list = eng.split("\n")
for line in eng_list:
    if re.match(r'^\[\[Category:.+\]\]$',line):
        # print(line)
        trg_str = re.match(r'^\[\[Category:(.*)\]\]$',line).group(1)
        # print(trg_str.group(1))
        # |*　<- split
        print(trg_str.split('|')[0])
        
# https://docs.python.org/ja/3/library/re.html 
# 正規表現にてかっこを用いるとその部分の文字列を抽出できる
# pattern.match.group(0) -> match全体
# pattern.match.group(1) -> まるかっこに対応するグループが順番に格納 

