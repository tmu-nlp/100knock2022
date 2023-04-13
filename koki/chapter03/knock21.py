import pandas as pd
import re #正規表現を扱うモジュール

df = pd.read_json('jawiki-country.json', lines = True)
uk = df.query('title == "イギリス"')['text'].values[0]

pattern = re.compile(r'\[\[(Category):(.*)\]\]') #compile関数は正規表現パターンオブジェクトを生成する、ちなみにraw文字列はエスケープシーケンスを無効化できる
#pattern = re.compile('Category') #テスト用
category_data = []
uk = uk.split('\n')#改行区切りで配列に直す

for text in uk:
    match = re.search(pattern, text) #見つからなければNoneを返す, match, search, findallの違いは下記マークダウン参照
    if match: 
        category_data.append(text) #パターンとマッチした行をまるごと抽出
    else:
        continue

[print(res) for res in category_data]
