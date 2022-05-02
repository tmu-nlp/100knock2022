#knock17
#1列目の文字列の種類（異なる文字列の集合）を求めよ．
# 確認にはcut, sort, uniqコマンドを用いよ．

import pandas as pd

#read_tableでテキストファイル読み込み，ヘッダーはなし
df = pd.read_table("popular-names.txt",header=None)

#iloc[行,列]は行，列番号指定で抽出する
#コロンで全行or全列指定
#drop_duplicate()で重複行削除
#sort_values()で一応ソートしとく
uniq_df = df.iloc[:,0].drop_duplicates().sort_values()
uniq_df.to_csv("output17-p.txt",sep='\t', index=False, header=False)
