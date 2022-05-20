import pandas as pd
import re

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
    match = re.sub('\'{2,}(.+?)\'{2,}', '\\1', t) # 文字列置換
    match2 = re.sub('\[\[(.+?)\]\]', '\\1', match)

    match3= re.sub('\[(.+?)\]', '\\1', match2) 
    match4 = re.sub('\*+(.+?)', '\\1', match3) 
    match5 = re.sub('\:+(.+?)', '\\1', match4)
    match6 = re.sub('\{\{(.+?)\}\}', '\\1', match5) 
    print(match6)
