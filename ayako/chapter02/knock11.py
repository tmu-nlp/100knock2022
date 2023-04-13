import pandas as pd

df = pd.read_table("popular-names.txt",header=None)

#セパレータは空白文字，インデックスとヘッダーは非表示で出力
df.to_csv("output11-p.txt",sep=' ', index=False, header=False)