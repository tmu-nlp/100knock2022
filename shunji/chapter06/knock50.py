"""
News Aggregator Data Setをダウンロードし、以下の要領で学習データ（train.txt），検証データ（valid.txt），評価データ（test.txt）を作成せよ．

ダウンロードしたzipファイルを解凍し，readme.txtの説明を読む．
情報源（publisher）が”Reuters”, “Huffington Post”, “Businessweek”, “Contactmusic.com”, “Daily Mail”の事例（記事）のみを抽出する．
抽出された事例をランダムに並び替える．
抽出された事例の80%を学習データ，残りの10%ずつを検証データと評価データに分割し，それぞれtrain.txt，valid.txt，test.txtというファイル名で保存する．
ファイルには，１行に１事例を書き出すこととし，カテゴリ名と記事見出しのタブ区切り形式とせよ（このファイルは後に問題70で再利用する）．
学習データと評価データを作成したら，各カテゴリの事例数を確認せよ．
"""


import pandas as pd
from sklearn.model_selection import train_test_split

# CSVの読込
df = pd.read_csv(
    "./NewsAggregatorDataset/newsCorpora.csv",
    header=None,
    sep="\t",
    names=[
        "ID",
        "TITLE",
        "URL",
        "PUBLISHER",
        "CATEGORY",
        "STORY",
        "HOSTNAME",
        "TIMESTAMP",
    ],
)

# publisherで絞ってcategoryとtitleだけのdfにする
df = df.loc[
    df["PUBLISHER"].isin(
        ["Reuters", "Huffington Post", "Businessweek", "Contactmusic.com", "Daily Mail"]
    ),
    ["TITLE", "CATEGORY"],
]

# データを訓練データ，テストデータ，評価データに分ける
train, valid_test = train_test_split(
    df,
    train_size=0.8,
    random_state=1,
    shuffle=True,
    stratify=df["CATEGORY"],  # stratifyはカテゴリの割合が各データで等しくなるように
)

valid, test = train_test_split(
    valid_test,
    test_size=0.5,
    random_state=1,
    shuffle=True,
    stratify=valid_test["CATEGORY"],
)

# データの保存
train.to_csv("./train.txt", sep="\t", index=False)
valid.to_csv("./valid.txt", sep="\t", index=False)
test.to_csv("./test.txt", sep="\t", index=False)

# 事例数の確認
if __name__ == "__main__":
    print("学習データ")
    print(train["CATEGORY"].value_counts())
    print("検証データ")
    print(valid["CATEGORY"].value_counts())
    print("評価データ")
    print(test["CATEGORY"].value_counts())
