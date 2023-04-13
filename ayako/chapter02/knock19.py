import pandas as pd

#read_tableでテキストファイル読み込み，ヘッダーはなし
df = pd.read_table("popular-names.txt",header=None)
#ユニークな要素とその出現回数の辞書を作成
df_dict = df.iloc[:,0].value_counts().to_dict()
#辞書をデータフレームに変換
freq_df = pd.DataFrame(list(df_dict.items()))
#出力
freq_df.to_csv("output19-p.txt",sep='\t', index=False, header=False)
