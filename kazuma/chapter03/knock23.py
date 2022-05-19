from knock20 import read_uk_text
import re

uk_text = read_uk_text()
uk_text = uk_text.split("\n")
ans = []
for line in uk_text:
    m = re.search(r"^(=+)(.+?)=+$", line)
    if m:
        ans.append(f"{len(m.group(1))-1}:{m.group(2).strip()}")
for ele in ans:
    print(ele)
