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
    if re.match(r'^\[\[ファイル:(.*)\]\]$',line):
        file = re.findall(r'^\[\[ファイル:(.*)\]\]$',line)
        trg_file = file[0].split('|')
        print(trg_file[0])
