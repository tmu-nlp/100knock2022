import pandas as pd
import json

df = pd.read_json('jawiki-country.json', lines = True)#lines引数により、jsonlを読み込む

#下記どちらでも良い
uk_articles = df[df['title']=='イギリス'].text.values[0]
#uk_articles = df.query('title == "イギリス"')['text'].values[0]

print(uk_articles)
