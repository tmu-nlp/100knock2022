import pandas as pd

wiki = pd.read_json("jawiki-country.json", lines=True) # lines=Trueで\nごと一行として取得
uk = wiki[wiki['title'] == 'イギリス']['text'].values[0] # ukのままだとndarray型

with open('uk.txt', 'w') as f:
  f.write(str(uk))
print(uk)