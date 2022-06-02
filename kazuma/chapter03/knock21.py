from knock20 import read_uk_text
import re

uk_text = read_uk_text()
uk_text = uk_text.split("\n")
ans = []
for line in uk_text:
    if re.search(r"^\[\[Category:.+\]\]$",line):
        ans.append(line)
for ele in ans:
    print(ele)