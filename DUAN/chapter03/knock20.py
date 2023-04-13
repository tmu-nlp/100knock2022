import pandas as pd
 
name = './100knock2022/DUAN/chapter03/jawiki-country.json'
data = pd.read_json(name, lines = True)
answ = data[data['title']=='イギリス']
an = answ['text'].values
print(an)
