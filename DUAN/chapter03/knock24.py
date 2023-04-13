import pandas as pd
import re
 
name = './100knock2022/DUAN/chapter03/jawiki-country.json'
data = pd.read_json(name, lines = True)
answ = data[data['title']=='イギリス']
an = answ['text'].values

media = re.findall('ファイル:(.+?)\|', an[0])
print('\n'.join(media))
