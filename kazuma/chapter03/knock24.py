from knock20 import read_uk_text
import re

uk_text = read_uk_text()
uk_text = uk_text.split("\n")
ans = []
for line in uk_text:
    m = re.search(r"\[\[ファイル:(.+?)(?:\|.+)?\]\]", line)
    if m:
        ans.append(m.group(1))
for ele in ans:
    print(ele)
