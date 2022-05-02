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
for line in eng_list:
    if re.match(r'^\=+.+\=+$',line):
        trg_str = re.match(r'^(\=+)(.+?)(\=+)$',line) #?をつけると最小matchとなる　
        name = trg_str.group(2).strip()
        level = len(trg_str.group(1))-1
        print(f'name : {name}, level : {level}')
