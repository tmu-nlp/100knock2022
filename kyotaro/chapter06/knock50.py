from email import header
import pandas as pd
import sklearn
from sklearn.model_selection import train_test_split

# ファイル読み込み
df = pd.read_csv("newsCorpora.csv", header=None, sep="\t", names=[
                 "ID", "TITLE", "URL", "PUBLISHER", "CATEGORY", "STORY", "HOSTNAME", "TIMESTAMP"])

# 抽出するPUBLISHERをまとめる
extract_publisher = ["Reuters", "Huffington Post",
                     "Businessweek", "Contactmusic.com", "Daily Mail"]

# 抽出するリストの中にPUBLISHERがいれば抽出
df = (df[df["PUBLISHER"].isin(extract_publisher)])

# 訓練データが8割、検証データとテストデータ合わせて2割 シャッフルはデフォルト
train, valid_test = train_test_split(
    df, train_size=0.8, random_state=1013, stratify=df["CATEGORY"])

# さらに検証データとテストデータに分ける　それぞれ半分づつ
valid, test = train_test_split(
    valid_test, train_size=0.5, random_state=1013, stratify=valid_test["CATEGORY"])

# テキストファイルにカテゴリとタイトルを書き込み
train.to_csv("train.txt", columns=[
             "CATEGORY", "TITLE"], sep='\t', index=False)
valid.to_csv("valid.txt", columns=[
             "CATEGORY", "TITLE"], sep='\t', index=False)
test.to_csv("test.txt", columns=["CATEGORY",
            "TITLE"], sep='\t', index=False)

# カテゴリ数確認
print("train-data")
print(train["CATEGORY"].value_counts())
print("\nvalid-data")
print(valid["CATEGORY"].value_counts())
print("\nvalue-data")
print(test["CATEGORY"].value_counts())

"""
wc -l newsCorpora.csv
  422937 newsCorpora.csv
"""
