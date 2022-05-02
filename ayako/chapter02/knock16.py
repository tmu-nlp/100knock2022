import pandas as pd

n = int(input())
#read_tableでテキストファイル読み込み，ヘッダーはなし
df = pd.read_table("popular-names.txt",header=None)

#n行ずつ分割してリストに格納
df_list = [df.iloc[i:i+n] for i in range(0,len(df),n)]

#リストの中のやつを一個ずつ出力
for i, df_i in enumerate(df_list):
    fname = "output16-p-"+str(i)+".txt"
    df_i.to_csv(fname,sep='\t', index=False, header=False)
