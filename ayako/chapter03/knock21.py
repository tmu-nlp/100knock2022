#knock21 カテゴリ名を含む行の抽出
import re
from knock20 import load_wiki

text = load_wiki("jawiki-country.json.gz").split("\n")
for line in text:
    if re.search(r'^\[\[Category:', line):
        print(line)












