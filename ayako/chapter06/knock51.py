# knock51
# 特徴量抽出
# 学習データ，検証データ，評価データから特徴量を抽出し，
# それぞれtrain.feature.txt，valid.feature.txt，test.feature.txtというファイル名で保存せよ． 
# なお，カテゴリ分類に有用そうな特徴量は各自で自由に設計せよ．
import pandas as pd
import re, pickle
from sklearn.feature_extraction.text import TfidfVectorizer
from nltk.tokenize import word_tokenize
from stemming.porter2 import stem

#前処理
def preprocessing(text):
    text = text.lower()#小文字化
    text = re.sub(r"\d+", "0", text)#数値表現は多様すぎて語頻度を下げてしまう->0など特定の数値に置き換える
    text = " ".join(word_tokenize(text))#単語分割（nltkのword_tokenizeはカンマや疑問符，短縮形とかもうまく分割してリスト返してくれる）
    text = stem(text)#ステミング
    return text

if __name__ == "__main__":
    #読み込んでDataFrameに変換
    train_df = pd.read_csv("train.txt", sep="\t", header=None)
    valid_df = pd.read_csv("valid.txt", sep="\t", header=None)
    test_df = pd.read_csv("test.txt", sep="\t", header=None)

    colums_name = ["TITLE", "CATEGORY"]
    train_df.columns = colums_name
    valid_df.columns = colums_name
    test_df.columns = colums_name

    #TITLEから予測を行うからTITLEの対して前処理
    #pandasオブジェクトの行や列に対して関数を適用するときはapply()
    train_df["TITLE"] = train_df["TITLE"].apply(preprocessing)
    valid_df["TITLE"] = valid_df["TITLE"].apply(preprocessing)
    test_df["TITLE"] = test_df["TITLE"].apply(preprocessing)
    
    #特徴量はTF-IDFを使う
    #TF:文書における指定単語の出現頻度
    #IDF:指定単語のレア度（逆文書頻度）
    #TF-IDF=TF*IDF
    #オプションのデフォルト:norm=’l2’, use_idf=True, smooth_idf=True, sublinear_tf=False
    vectorizer = TfidfVectorizer()
    #訓練データから五位の獲得，idfの計算
    #ココッテ開発データは一緒にするべきなの？？？
    vectorizer.fit(train_df["TITLE"])
    #fitで得た語彙からtf-idf行列に変換
    train_tfidf = vectorizer.transform(train_df["TITLE"])
    valid_tfidf = vectorizer.transform(valid_df["TITLE"])
    test_tfidf = vectorizer.transform(test_df["TITLE"])

    #リストぽくしたいからtoarray()で変形
    train_tfidf = train_tfidf.toarray()
    valid_tfidf = valid_tfidf.toarray()
    test_tfidf = test_tfidf.toarray()

    #それぞれDataframeに変換してから保存
    pd.DataFrame(data=train_tfidf).to_csv("train.feature.txt", sep="\t", index=False, header=None)
    pd.DataFrame(data=valid_tfidf).to_csv("valid.feature.txt", sep="\t", index=False, header=None)
    pd.DataFrame(data=test_tfidf).to_csv("test.feature.txt", sep="\t", index=False, header=None)
    
    #knock57用に[単語]=単語idになる辞書をvocablary_から取得
    with open("vocabulary.pkl", "wb") as f:
        pickle.dump(vectorizer.vocabulary_, f)