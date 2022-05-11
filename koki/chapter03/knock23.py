import pandas as pd
import re #正規表現を扱うモジュール

df = pd.read_json('jawiki-country.json', lines = True)#JSON Lineの読み込み
uk = df.query('title == "イギリス"')['text'].values[0]
uk = uk.split('\n')#改行区切りで配列に直す

pattern = re.compile(r'^(=+)(.*?)(=+)$') 
#「=+」で=の２回以上の繰り返し、「.*」で任意文字列の0回以上の繰り返しを表す
#「^」は文字列の先頭、「$」は文字列の終端を表す
#「?」は直前の文字が0回または1回の繰り返しを表す

section_name = []
section_level = []

for text in uk:
    match = re.search(pattern, text)
    if match:
        section_level.append(len(match.group(1)))#セクションレベルを格納 (「=」の数を格納)
        section_name.append(match.group(2))
        #section_name.append(match.group(2).strip('=')) #=を除去してセクション名のみを格納
        #セクション名の後の「=」を正規表現で認識するいい方法がわからなかった→末尾に「?」
    else:
        continue

df = pd.DataFrame(list(zip(section_name, section_level)), columns=['Name', 'Level'])
print(df)
