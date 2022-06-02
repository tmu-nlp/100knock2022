import pandas as pd
import re
import requests

# ファイルの読み込み 
name = './100knock2022/DUAN/chapter03/jawiki-country.json'
data = pd.read_json(name, lines = True)
answ = data[data['title']=='イギリス']
an = answ['text'].values

d = {} 
for t in an[0].split('\n'): # 要素を改行文字で区切る
    # re.search関数は文字列にパターンとマッチする部分があるかを調べる
    if re.search('\|(.+?)\s=\s*(.+)', t): 
        tem = re.search('\|(.+?)\s*=\s*(.+)', t)
        d[tem[1]] = tem[2] # キーと値を指定し、dに追加する

S = requests.Session()
U = 'https://en.wikipedia.org/w/api.php'
 
P = {'action':'query', 'format':'json', 'prop':'imageinfo', 'titles':f"File:{d['国旗画像']}", 'iiprop':'url'}
R = S.get(url=U, params=P)
data = R.json()
page = data['query']['pages']

for k, v in page.items():
    print(v['imageinfo'][0]['url'])
