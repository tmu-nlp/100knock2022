#knock27 内部リンクの除去
import re
from knock25 import template

result = {}
for line in template[0][0].split("\n"): #templateリストの中にタプルがある
    if re.search(r"^\|.+?\s=\s*",line):
        field = re.findall(r"^\|(.+?)\s*=\s*(.+)",line)
        remove_quote = re.sub(r"\'{2,5}",r"",field[0][1]) #'を除去
        remove_markup = re.sub(r"\[\[(?:[^|]+?\|)?(.+?)\]\]",r"\1",remove_quote)
        result[field[0][0]] = remove_markup

for key, value in result.items():
    print(key,"\t",value)