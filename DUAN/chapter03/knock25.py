# 25. テンプレートの抽出Permalink
# 記事中に含まれる「基礎情報」テンプレートのフィールド名と値を抽出し，辞書オブジェクトとして格納せよ．

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
