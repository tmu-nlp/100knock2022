import gzip
import json
f = gzip.open('jawiki-country.json.gz', mode='rt')
f_uk = open('jawiki-uk.txt', 'w')
for line in f:
    obj = json.loads(line) #json.loads()でJSON文字列を辞書に変換
    if (obj['title']) == 'イギリス':
        f_uk.write(obj['text']) 
f_uk.close()
