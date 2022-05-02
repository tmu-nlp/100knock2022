import pandas as pd

#read_tableでテキストファイル読み込み，ヘッダーはなし
df = pd.read_table("popular-names.txt",header=None)
#第一引数はソート対象の列，第二引数は降順の時False
sorted_df = df.sort_values(2,ascending=False)
#出力
sorted_df.to_csv("output18-p.txt",sep='\t', index=False, header=False)
