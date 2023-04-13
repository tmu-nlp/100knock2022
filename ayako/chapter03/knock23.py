#knock23 セクション構造
import re
from knock20 import load_wiki

text = load_wiki("jawiki-country.json.gz").split("\n")
for line in text:
    if re.search(r"^={2,}(.+?)={2,}", line):
        section = re.findall(r"^(={2,})\s*(.+?)\s*(={2,})",line)
        level = len(section[0][0]) - 1 #section[0][0] = "="の数
        print(section[0][1],"level:",level)