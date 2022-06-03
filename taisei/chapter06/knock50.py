import pandas as pd
from sklearn.model_selection import train_test_split
"""
FORMAT: ID \t TITLE \t URL \t PUBLISHER \t CATEGORY \t STORY \t HOSTNAME \t TIMESTAMP
"""

publisher_list = ["Reuters", "Huffington Post", "Businessweek", "Contactmusic.com", "Daily Mail"]

df = pd.read_table("./NewsAggregatorDataset/newsCorpora.csv", names=("ID", "TITLE", "URL", "PUBLISHER", "CATEGORY", "STORY", "HOSTNAME", "TIMESTAMP"))

#事例抽出
df_extract = df.query(f'PUBLISHER in {publisher_list}') 

#データ分割　各カテゴリのCATEGORY事例数の割合が等しくなるように分類 再現できるように乱数シード固定
df_train, df_valid_test = train_test_split(df_extract, test_size=0.2, stratify=df_extract["CATEGORY"], random_state=0) #学習データ(80%)と検証・評価データ(20%)に分割
df_valid, df_test = train_test_split(df_valid_test, test_size=0.5, stratify=df_valid_test["CATEGORY"], random_state=0) #検証・評価データを検証データ(10%)と評価データ(10%)に分割

#各データのCATEGORYとTITLEのみ選択
df_train_ct = df_train.loc[:, ["CATEGORY", "TITLE"]]
df_valid_ct = df_valid.loc[:, ["CATEGORY", "TITLE"]]
df_test_ct = df_test.loc[:, ["CATEGORY", "TITLE"]]

if __name__ == "__main__":
    #各データをファイルに書き出す
    df_train_ct.to_csv("./output/train.txt", sep="\t", index=False, header=None) 
    df_valid_ct.to_csv("./output/valid.txt", sep="\t", index=False, header=None)
    df_test_ct.to_csv("./output/test.txt", sep="\t", index=False, header=None)

    #各カテゴリの事例数
    print(df_train_ct["CATEGORY"].value_counts()) #b:4502  e:4223  t:1219  m:728
    print(df_valid_ct["CATEGORY"].value_counts()) #b:562  e:528  t:153  m:91
    print(df_test_ct["CATEGORY"].value_counts()) #b:563  e:528  t:152  m:91

