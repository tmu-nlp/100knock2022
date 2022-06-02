import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
import re
import pickle
"""
FORMAT: ID \t TITLE \t URL \t PUBLISHER \t CATEGORY \t STORY \t HOSTNAME \t TIMESTAMP
"""

#前処理
def pretreat(text):
    #小文字化
    text_lower = text.lower()

    #英文字、数字、スペース以外を削除
    text_alnum = re.sub(r'[^a-z0-9 ]', r'', text_lower)

    #数字は数字トークンとして扱うために全て0に置換
    text_al0 = re.sub(r'[0-9]+', r'0', text_alnum)

    return text_al0


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

# TITLEの要素に関数pretreatを適用
df_train_ct["TITLE"] = df_train_ct["TITLE"].apply(pretreat)
df_valid_ct["TITLE"] = df_valid_ct["TITLE"].apply(pretreat)
df_test_ct["TITLE"] = df_test_ct["TITLE"].apply(pretreat)

# tf-idf
vectorizer = TfidfVectorizer(use_idf=True, smooth_idf=True, ngram_range=(1, 1))
#訓練データのTITLEに基づいて語彙の獲得やidfを計算
vectorizer.fit(df_train_ct["TITLE"]) 

#↑（訓練データのtf-idf）をもとに各データのタイトルをtf-idf行列に変換
tfidf_train = vectorizer.transform(df_train_ct["TITLE"])
tfidf_valid = vectorizer.transform(df_valid_ct["TITLE"])
tfidf_test = vectorizer.transform(df_test_ct["TITLE"])

df_tfidf_train = pd.DataFrame(data=tfidf_train.toarray(), columns=vectorizer.get_feature_names())
df_tfidf_valid = pd.DataFrame(data=tfidf_valid.toarray(), columns=vectorizer.get_feature_names())
df_tfidf_test = pd.DataFrame(data=tfidf_test.toarray(), columns=vectorizer.get_feature_names())
voc = vectorizer.vocabulary_ #語彙を取得 57で使う（中身は、単語,その単語の重み内でのインデックス番号　が各単語分）

if __name__ == "__main__":
    df_tfidf_train.to_csv("./output/train.feature.txt", sep="\t", index=False) 
    df_tfidf_valid.to_csv("./output/valid.feature.txt", sep="\t", index=False) 
    df_tfidf_test.to_csv("./output/test.feature.txt", sep="\t", index=False)
    with open("./output/knock51_lr_voc", "wb") as f_out:
        pickle.dump(voc, f_out)