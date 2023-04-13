#knock28 マークアップの除去
import re
from knock25 import template

result = {}
for line in template[0][0].split("\n"): #templateリストの中にタプルがある
    if re.search(r"^\|.+?\s=\s*",line):
        field = re.findall(r"^\|(.+?)\s*=\s*(.+)",line)
        remove_quote = re.sub(r"\'{2,5}",r"",field[0][1]) #シングルクオートを除去
        remove_lang = re.sub(r"\{\{(?:lang|仮リンク)(?:[^|]*?\|)*?([^|]*?)\}\}",r"\1",remove_quote) #langのとこ除去
        remove_markup = re.sub(r"\[\[(?:[^|]+?\|)?(.+?)\]\]",r"\1",remove_lang)#内部マークアップ除去
        remove_ref = re.sub(r"<.+?>",r"",remove_markup)#参照を除去
        remove_http = re.sub(r"\[http.+?\]",r"",remove_ref)#http，httpsを除去
        result[field[0][0]] = remove_http

if __name__ == "__main__":
    for key, value in result.items():
        print(key,"\t",value)