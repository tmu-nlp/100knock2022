#!/usr/bin/python3

import json
import gzip
import re

with gzip.open('jawiki-country.json.gz', 'rt') as f:
    for line in f:
        content = json.loads(line) # type-str so s is added
        if ("title","イギリス") in content.items():
            eng = content["text"]

# print(eng)

eng_list = eng.split("\n")
# print(eng_list)

# |hoge = huga を認識できるように
pattern = re.compile('\|(.+?)\s=\s*(.+)')

trg_dict = dict()

# remove '' , ''' , '''''
for line in eng_list:
    line_match = re.match(pattern,line)
    if line_match:
        trg_line = line_match.group(0)
        midput = re.sub('\'+','',trg_line) # 1個以上の'を削除
        # print(midput)
        # print(trg_line)
        output = re.match(pattern,midput)
        trg_dict[output.group(1)] = output.group(2)

for key,value in trg_dict.items():
    print(f'{key} : {value}')