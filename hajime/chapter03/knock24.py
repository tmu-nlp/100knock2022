#!/usr/bin/python3
'''
24. ファイル参照の抽出Permalink
記事から参照されているメディアファイルをすべて抜き出せ．
'''

import json
import gzip
import re

# イギリスの記事を獲得
with gzip.open('jawiki-country.json.gz', 'rt') as f:
    for line in f:
        content = json.loads(line)  # type-str so s is added
        if ("title", "イギリス") in content.items():
            eng = content["text"]

eng_list = eng.split("\n")
for line in eng_list:
    # [[ファイル:hoge.png|thumb|説明文]]となる部分を調査
    if re.match(r'^\[\[ファイル:(.*)\]\]$', line):
        # 'ファイル:hoge.png|thumb|説明文'を変数fileに格納
        file = re.findall(r'^\[\[ファイル:(.*)\]\]$', line)
        trg_file = file[0].split('|')
        print(trg_file[0])
