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

# 一行読み込んで対象部分を全て処理する

def remove_markup(text):
    inlink_pattern = r'\[\[(?:[^|]*?\|)??([^|]*?)\]\]'
    text = re.sub(inlink_pattern,r'\1', text)
    outlink_pattern = r'https?://[\w/:%#\$&\?\(\)~\.=\+\-…]+'
    text = re.sub(outlink_pattern, '', text)
    html_pattern = r'\<(.+?)\>'
    text = re.sub(html_pattern, '', text)
    brace_pattern =  r'\{\{(.+\||)(.+?)\}\}'
    text = re.sub(brace_pattern, '', text)
    except_pattern = r'\[\[(女王陛下万歳)\|'
    text = re.sub(except_pattern, r'\1', text)
    return text

result = {key: remove_markup(value) for key, value in trg_dict.items()}

for key, value in result.items():
    print(key + ": " + value)