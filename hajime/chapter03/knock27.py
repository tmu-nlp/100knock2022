#!/usr/bin/python3

import json
import gzip
import re

with gzip.open('jawiki-country.json.gz', 'rt') as f:
    for line in f:
        content = json.loads(line) # type-str so s is added
        if ("title","イギリス") in content.items():
            eng = content["text"]

eng_list = eng.split("\n")

# |hoge = huga を認識できるように
pattern = re.compile('\|(.+?)\s=\s*(.+)')
trg_dict = dict()

# remove ('.+)
for line in eng_list:
    line_match = re.match(pattern,line)
    if line_match:
        trg_line = line_match.group(0)
        midput = re.sub('\'+','',trg_line) # 1個以上の'を削除
        output = re.match(pattern,midput)
        trg_dict[output.group(1)] = output.group(2)

# ここで sub [[記事名]] -> 記事名, [[記事名|表示名]]->表示名, [[記事名#節名|表示名]]->表示名 の処理
# 一行読み込んで対象部分を全て処理する

def remove_markup(text):
    link_pattern = r'\[\[(?:[^|]*?\|)??([^|]*?)\]\]'
    text = re.sub(link_pattern,r'\1', text)
    # r'\1' はgroup(1)に対応
    # https://docs.pyq.jp/python/library/re.html
    return text

result = {key: remove_markup(value) for key, value in trg_dict.items()}

for key, value in result.items():
    print(key + ": " + value)
