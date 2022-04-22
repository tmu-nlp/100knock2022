#knock12
#各行の1列目だけを抜き出したものをcol1.txtに，
#2列目だけを抜き出したものをcol2.txtとしてファイルに保存せよ．
#確認にはcutコマンドを用いよ．

import pandas as pd

#read_tableでテキストファイル読み込み，ヘッダーはなし
df = pd.read_table("popular-names.txt",header=None)

#セパレータはタブ，インデックスとヘッダーは非表示で出力
#iloc[行,列]は行，列番号指定で抽出する
#コロンで全行or全列指定
df.iloc[:,0].to_csv("col1-p.txt",sep='\t', index=False, header=False)
df.iloc[:,1].to_csv("col2-p.txt",sep='\t', index=False, header=False)