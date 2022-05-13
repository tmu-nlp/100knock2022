import pandas as pd
import re

# ファイルの読み込み 
name = "jawiki-country.json"
data = pd.read_json(name, lines = True)
answ = data[data["title"]=="イギリス"]
an = answ["text"].values

d = {} 
for t in an[0].split("\n"): # 要素を改行文字で区切る

    # re.search関数は文字列にパターンとマッチする部分があるかを調べる
    if re.search("\|(.+?)\s=\s*(.+)", t): 
        tem = re.search("\|(.+?)\s*=\s*(.+)", t)
        d[tem[1]] = tem[2] # キーと値を指定し、dに追加する
print(d)
