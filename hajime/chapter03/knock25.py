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

for line in eng_list:
    trg_line = re.match(pattern,line)
    if trg_line:
        # print(trg_line.group(1))
        trg_dict[trg_line.group(1)] = trg_line.group(2)

# print(trg_dict)
for key,value in trg_dict.items():
    print(f'{key} : {value}')

# print(trg_dict)

# json -> {{基礎情報.*?}} -> split('|') -> dict
# regular expression?

# if re.findall('\{\{基礎情報.*\}\}',eng):
#     print(re.sub(r'^\{\{基礎情報.*\}\}$',eng))
# return none

# for line in eng_list:
#     if re.match(r'^\{\{基礎情報.*\}\}$',line):
#         print(line)
# return none 