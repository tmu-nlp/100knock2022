'''
51. 特徴量抽出
特徴量抽出して、train.feature.txt，valid.feature.txt，test.feature.txtというファイル名で保存
カテゴリ分類に有用そうな特徴量: word frequency or bi-gram word frequency(数字や句読点は入れない)
記事の見出しを単語列に変換したものが最低限のベースライン
process:
pre-processing: lowercase and remove all punctuation, set numbers to zero
applying TfidfVectorizer to get TF-IDF feature matrix


tools:
1. sklearn.feature_extraction.text.TfidfVectorizer :
    object :Convert a collection of raw documents to a matrix of TF-IDF features.
    document:https://scikit-learn.org/stable/modules/generated/sklearn.feature_extraction.text.TfidfVectorizer.html#sklearn.feature_extraction.text.TfidfVectorizer
'''

from knock50 import get_data
import string
import re
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer


def pre_processing(doc):
    # doc == sentence == title
    trantab = str.maketrans(string.punctuation, ' '*len(string.punctuation))
    doc = doc.translate(trantab)
    doc = doc.lower()
    doc = re.sub(r'[0-9]+', '0', doc)
    return doc

def df_pre(train, valid, test):
    # concatenate data and reset index
    df = pd.concat([train, valid, test], axis=0)   # 列軸で
    df.reset_index(drop=True, inplace=True)      # 重新设置索引后，删除原索引；在原df上修改
    # apply pre-processing:
        # df.map(arg, na_action=None)：accept func or dict as input, mapping correspondence and return df or series with same as index as caller

    df['TITLE'] = df['TITLE'].map(lambda x: pre_processing(x))
    return df

def get_features(train, valid, test):
    # vec_tfidf = TfidfVectorizer()
        # default setting:token_pattern='(?u)\b\w\w+\b',ngram_range(1,1),max_df=1.0,min_df=1, norm='l2', use_idf=True, smooth_idf=True
    vec_tfidf = TfidfVectorizer(ngram_range=(1, 2), min_df=5)

    # ベクトル化
    X_train = vec_tfidf.fit_transform(train['TITLE']).toarray()
    columns = vec_tfidf.get_feature_names()
    # print(columns[:19])
    X_train = pd.DataFrame(X_train, columns=columns)

    X_valid = vec_tfidf.fit_transform(valid['TITLE']).toarray()
    columns = vec_tfidf.get_feature_names()
    X_valid = pd.DataFrame(X_valid, columns=columns)

    X_test = vec_tfidf.fit_transform(test['TITLE']).toarray()
    columns = vec_tfidf.get_feature_names()
    X_test = pd.DataFrame(X_test, columns=columns)

    return (X_train, X_valid, X_test)


if __name__ == '__main__':
    file_path = '../data/newsCorpora.csv'
    data = get_data(file_path)
    # read raw texts from knock50
    train = pd.read_table('./train.txt', names=['CATEGORY', 'TITLE'])
    valid = pd.read_table('./valid.txt', names=['CATEGORY', 'TITLE'])
    test = pd.read_table('./test.txt', names=['CATEGORY', 'TITLE'])
    df_pre = df_pre(train, valid, test)

    train_re, val_test_re = train_test_split(df_pre, test_size=0.2, shuffle=True, random_state=886,
                                       stratify=df_pre['CATEGORY'])
    valid_re, test_re = train_test_split(val_test_re, test_size=0.5, shuffle=True, random_state=886,
                                   stratify=val_test_re['CATEGORY'])

    train_re.to_csv('./train_re.txt', columns=['CATEGORY', 'TITLE'], sep='\t', header=False, index=False)
    valid_re.to_csv('./valid_re.txt', columns=['CATEGORY', 'TITLE'], sep='\t', header=False, index=False)
    test_re.to_csv('./test_re.txt', columns=['CATEGORY', 'TITLE'], sep='\t', header=False, index=False)

    X_train = get_features(train_re, valid_re, test_re)[0]
    X_valid = get_features(train_re, valid_re, test_re)[1]
    X_test = get_features(train_re, valid_re, test_re)[2]

    print(X_train.shape, '\t', X_valid.shape, '\t', X_test.shape)
    # (10672, 5625)  (1334, 588)  (1334, 602)

    X_train.to_csv('./X_trian_features.txt', sep='\t', index=False)
    X_valid.to_csv('./X_valid_features.txt', sep='\t', index=False)
    X_test.to_csv('./X_test_features.txt', sep='\t', index=False)











