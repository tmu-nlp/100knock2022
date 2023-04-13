import pandas as pd
import re #正規表現を扱うモジュール

df = pd.read_json('jawiki-country.json', lines = True)
uk = df.query('title == "イギリス"')['text'].values[0]

#方針 :インデックス名とカテゴリ名をそれぞれグループ化してとりだす
pattern = re.compile(r'\[\[(Category):(.*)\]\]') #compile関数は正規表現パターンオブジェクトを生成する、raw文字列はエスケープシーケンスを無効化できる
category_name = []
uk = uk.split('\n')#改行区切りで配列に直す

for text in uk:
    match = re.search(pattern, text) #見つからなければNoneを返す, match, search, findallの違いは下記マークダウン参照
    if match: 
        tmp = re.sub(r'\|.*', '', match.group(2))#sub...置換, パイプなど不要な文字を除去
        category_name.append(tmp)
    else:
        continue


[print(res) for res in category_name]
