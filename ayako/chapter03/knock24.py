#knock24 ファイル参照の抽出
import re
from knock20 import load_wiki

text = load_wiki("jawiki-country.json.gz").split("\n")
for line in text:
    if re.search(r"^\[\[ファイル:(.+?)", line):
        print(re.findall(r"(?:ファイル:)(.+?)\|",line))