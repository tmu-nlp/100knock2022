import pandas as pd
import re #正規表現を扱うモジュール

df = pd.read_json('jawiki-country.json', lines = True)
uk = df.query('title == "イギリス"')['text'].values[0]
uk = uk.split('\n')#改行区切りで配列に直す

def remove_mediawiki_markup(text):
    remove_pattern_emph = re.compile(r'\'{1,5}') #' の1~5回の繰り返しパターン
    remove_pattern_link = re.compile(r'\[\[(.*?)\]\]') #?により直前の文字が繰り返すかマッチ
    remove_pattern_tag = re.compile(r'\<(.*?)\>') #タグのパターン
    remove_pattern_url = re.compile(r'\[http.?://.*?\]') #urlのパターン
    remove_pattern_lang = re.compile(r'\{\{(.*?)\}\}') #template:langのパターン

    if re.search(remove_pattern_emph, text):
        text = re.sub(remove_pattern_emph, '', text)

    if re.search(remove_pattern_link, text):
        content = re.search(remove_pattern_link, text)
        text = re.sub(remove_pattern_link, content.group(1), text) #リンクの内容は消さないように注意

    if re.search(remove_pattern_tag, text):
        text = re.sub(remove_pattern_tag, '', text)

    if re.search(remove_pattern_url, text):
        text = re.sub(remove_pattern_url, '', text)     

    if re.search(remove_pattern_lang, text):
        text = re.sub(remove_pattern_lang, '', text)    
    
    return text

pattern_template = re.compile(r'\|(.*?)\s=\s*(.*)')# 「|(key) = (value)」を認識
res_dict = {}

for text in uk:
    match = re.search(pattern_template, text)
    if match:
        text = remove_mediawiki_markup(match.group(2))#除去処理
        res_dict[match.group(1)] = text
    
#[print(key, value, sep = ': ') for key, value in res_dict.items()]
