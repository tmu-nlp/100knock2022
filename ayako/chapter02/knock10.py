import pandas as pd

#read_tableでテキストファイル読み込み，ヘッダーはなし
df = pd.read_table("popular-names.txt",header=None)

#セパレータはタブ，インデックスとヘッダーは非表示で出力
df.to_csv("output10-p.txt",sep='\t', index=False, header=False)