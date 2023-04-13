#knock25 テンプレートの抽出
import re
from knock20 import load_wiki

text = load_wiki("jawiki-country.json.gz")
result = {}
template = re.findall(r"^\{\{基礎情報.*?$(.*?)^(\}\}$)", text, re.MULTILINE + re.DOTALL)
for line in template[0][0].split("\n"):
    if re.search(r"^\|.+?\s*=\s*",line):
        field = re.findall(r"^\|(.+?)\s*=\s*(.+)",line)
        result[field[0][0]] = field[0][1]
if __name__ == "__main__":
    for key, value in result.items():
        print(key,"\t",value)