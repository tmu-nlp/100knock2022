import pandas as pd

n = int(input())
#read_tableでテキストファイル読み込み，ヘッダーはなし
df = pd.read_table("popular-names.txt",header=None)

#セパレータはタブ，インデックスとヘッダーは非表示で出力
#iloc[行,列]は行，列番号指定で抽出する
#コロンで全行or全列指定
df.iloc[0:n].to_csv("output14-p.txt",sep='\t', index=False, header=False)
