import pandas as pd

#read_tableでテキストファイル読み込み，ヘッダーはなし
df1 = pd.read_table("col1-p.txt",header=None)
df2 = pd.read_table("col2-p.txt",header=None)

#concat関数でデータ結合
#axis=1で横への連結
df_concat = pd.concat([df1,df2],axis=1)
df_concat.to_csv("output13-p.txt",sep='\t', index=False, header=False)