import pandas as pd
import re
 
name = './100knock2022/DUAN/chapter03/jawiki-country.json'
data = pd.read_json(name, lines = True)
answ = data[data['title']=='イギリス']
an = answ['text'].values

for t in an[0].split('\n'):
    if re.search('Category', t):
        t = t.replace('[[Category:', '').replace('|*]]', '').replace(']]', '')
        print(t)
