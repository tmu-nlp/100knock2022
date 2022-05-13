#!/usr/bin/python3

import json
import gzip

with gzip.open('jawiki-country.json.gz', 'rt') as f:
    for line in f:
        content = json.loads(line)  # type-str so s is added
        if ("title", "イギリス") in content.items():
            # print(content)
            print(content["text"])
