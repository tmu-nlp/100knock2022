#knock26 強調マークアップの除去
#25の処理時に，テンプレートの値からMediaWikiの強調マークアップ
#（弱い強調，強調，強い強調のすべて）を除去してテキストに変換せよ
import re
from knock25 import template

result = {}
for line in template[0][0].split("\n"): #templateリストの中にタプルがある
    if re.search(r"^\|.+?\s=\s*",line):
        field = re.findall(r"^\|(.+?)\s*=\s*(.+)",line)
        remove_quote = re.sub(r"\'{2,5}",r"",field[0][1]) #'を除去
        result[field[0][0]] = remove_quote

for key, value in result.items():
    print(key,"\t",value)