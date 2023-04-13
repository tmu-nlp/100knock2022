#knock22 カテゴリ名の抽出
import re
from knock20 import load_wiki

text = load_wiki("jawiki-country.json.gz").split("\n")
for line in text:
    if re.search(r"^\[\[Category:", line):
        print(re.findall(r"\[\[Category:(.*?)(?:\|.*)?\]\]",line))