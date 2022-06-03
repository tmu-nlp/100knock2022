import pandas as pd
import string
import re
from sklearn.feature_extraction.text import TfidfVectorizer
from knock50 import train_data, valid_data, test_data

def preprosessing(text):
    '''前処理'''
    table = str.maketrans(string.punctuation, ' '*len(string.punctuation))
    text = text.translate(table)
    text = text.lower()
    pattern = re.compile('[0-9]+')
    text = re.sub(pattern, '0', text)

    return text

#データの連結、前処理
df = pd.concat([train_data, valid_data, test_data], axis = 0)
df.reset_index(drop=True, inplace=True)
df['TITLE'] = df['TITLE'].map(lambda x: preprosessing(x)) #map関数を使ってSeriesの各要素に前処理の関数を適用

#単語のベクトル化
vec_tfidf = TfidfVectorizer() #TfidfVectorizerのインスタンス生成
data = vec_tfidf.fit_transform(df['TITLE'])
data = pd.DataFrame(data.toarray(), columns = vec_tfidf.get_feature_names_out())

#分割幅の指定
split_point1 = int(len(data)//3)
split_point2 = int(split_point1 * 2)

#学習、検証、評価データ
x_train = data[:split_point1]
x_valid = data[split_point1:split_point2]
x_test = data[split_point2:]

#学習、検証、評価等別
y_data = df['CATEGORY']
y_train = y_data[:split_point1]
y_valid = y_data[split_point1:split_point2]
y_test = y_data[split_point2:]

#特徴量の書き出し
x_train.to_csv('train.feature.txt', sep = '\t', index = False)
x_valid.to_csv('valid.feature.txt', sep = '\t', index = False)
x_test.to_csv('test.feature.txt', sep = '\t', index = False)
