#!/usr/bin/python3

import json
import gzip

with gzip.open('jawiki-country.json.gz', 'rt') as f:
    for line in f:
        content = json.loads(line) # type-str so s is added
        if ("title","イギリス") in content.items():
            eng = content["text"]

# print(type(eng))
eng_list = eng.split("\n")
# print(eng_list)
for line in eng_list:
    if "[Category" in line:
        print(line)

