import pandas as pd
import re #正規表現を扱うモジュール

df = pd.read_json('jawiki-country.json', lines = True)
uk = df.query('title == "イギリス"')['text'].values[0]
uk = uk.split('\n')#改行区切りで配列に直す

#pattern = re.compile(r'ファイル:(.*)') #ファイル: (ファイル名) | thumb | (説明文)
pattern = re.compile(r'ファイル:(.*)\|thumb\|.*\|(.*)') #ファイル: (ファイル名) | thumb | leftなど, ここが謎 | (説明文)

file_name = []
file_content = []

for text in uk:
    match = re.search(pattern, text)
    if match:
        file_name.append(match.group(1))
        file_content.append(match.group(2))
    else:
        continue

df = pd.DataFrame(list(zip(file_name, file_content)), columns=['File Name', 'Content'])
#print(len(df)) #読み込み確認
print(df.iloc[:,0])
df.to_csv('output24.csv', sep = ',', index=None)
