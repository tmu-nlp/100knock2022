import pandas as pd
import re #正規表現を扱うモジュール

df = pd.read_json('jawiki-country.json', lines = True)
uk = df.query('title == "イギリス"')['text'].values[0]
uk = uk.split('\n')#改行区切りで配列に直す

pattern = re.compile(r'\|(.*)\s=\s*(.*)')# 「|(key) = (value)」を認識, \sは空白を表す

template_dict = {}

for text in uk:
    match = re.search(pattern, text)
    if match:
        template_dict[match.group(1)] = match.group(2)
    else:
        continue

[print(key, value, sep = ':\t') for key, value in template_dict.items()]
