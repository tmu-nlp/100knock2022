'''
51. 特徴量抽出
特徴量抽出して、train.feature.txt，valid.feature.txt，test.feature.txtというファイル名で保存
カテゴリ分類に有用そうな特徴量: word frequency or bi-gram word frequency(数字や句読点は入れない)
記事の見出しを単語列に変換したものが最低限のベースライン

process:
pre-processing: lowercase and remove all punctuation, set numbers to zero
get word vectors

tools:
1. class TfidfVectorizer() == CountVectorizer(sparse representation of token cuonts) + TfidfTransformer(convert counts to normalized tf-idf representation)
    to Convert a collection of raw docs to a matrix of TF-IDF features. == CounterVectorizer followed by TfidfTransformer
    methods:
    refer_blog_cn:https://www.thinbug.com/q/53027864
    .fit(raw_doc,y=None): 拟合训练数据，并保存到vectorizer变量中。learn the vocab of vectorizer, return fitted vectorizer.
    .transform(list of docs).toarray(): 使用fit()的变量输出来转换val/test数据。apply vetorizer to new sentence and output in array form.
    .fit_transformer(raw_doc,y=None) == fit+transform: 拟合后直接转换数据。return sparse matrix of (n_samps, n_feats) contains Tf-idf-weighted doc-term matrix for re-using.

documentation:https://scikit-learn.org/stable/modules/generated/sklearn.feature_extraction.text.TfidfVectorizer.html#sklearn.feature_extraction.text.TfidfVectorizer
'''

import string
import re
import pandas as pd

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
    # print(df.shape)  -> (13340, 2)
    df.reset_index(drop=True, inplace=True)      # 重新设置索引后，删除原索引；在原df上修改
    # apply pre-processing:
        # df.map(arg, na_action=None)：accept func or dict as input, mapping correspondence and return df or series with same as index as caller

    df['TITLE'] = df['TITLE'].map(lambda x: pre_processing(x))
    return df

def get_features(train_valid, test):
    # make vectorizer
        # default setting:token_pattern='(?u)\b\w\w+\b',ngram_range(1,1),max_df=1.0,min_df=1, norm='l2', use_idf=True, smooth_idf=True
    vec_tfidf = TfidfVectorizer(ngram_range=(1, 2), min_df=10)

    # ベクトル化
    X_train_valid = vec_tfidf.fit_transform(train_valid['TITLE']).toarray()
    columns = vec_tfidf.get_feature_names()
    # print(columns[:19])
    X_train_valid = pd.DataFrame(X_train_valid, columns=columns)

    X_test = vec_tfidf.transform(test['TITLE']).toarray()
    columns = vec_tfidf.get_feature_names()
    X_test = pd.DataFrame(X_test, columns=columns)

    return (X_train_valid, X_test)


if __name__ == '__main__':
    # read raw texts from knock50
    train = pd.read_table('./train.txt', names=['CATEGORY', 'TITLE'])
    valid = pd.read_table('./valid.txt', names=['CATEGORY', 'TITLE'])
    test = pd.read_table('./test.txt', names=['CATEGORY', 'TITLE'])
    # get preprocessed df data
    df_pre = df_pre(train, valid, test)
    train_valid_re = df_pre[:len(train) + len(valid)]
    train_re = train_valid_re[:len(train)]
    valid_re = train_valid_re[len(train):]
    test_re = df_pre[len(train) + len(valid):]
    # print(train_re.shape, '\t', valid_re.shape, '\t', test_re.shape)
    # (10672, 2) 	 (1334, 2) 	 (1334, 2)

    train_re.to_csv('./train_re.txt', sep='\t', header=False, index=False)
    valid_re.to_csv('./valid_re.txt', sep='\t', header=False, index=False)
    test_re.to_csv('./test_re.txt', sep='\t', header=False, index=False)

    X_train = get_features(train_valid_re, test_re)[0][:len(train)]
    X_valid = get_features(train_valid_re, test_re)[0][len(train):]
    X_test = get_features(train_valid_re, test_re)[1]

    # print(X_train.shape, '\t', X_valid.shape, '\t', X_test.shape)
    # (10672, 2814) (1334, 2814) (1334, 2814)

    X_train.to_csv('./X_train_features.txt', sep='\t', index=False)
    X_valid.to_csv('./X_valid_features.txt', sep='\t',  index=False)
    X_test.to_csv('./X_test_features.txt', sep='\t', index=False)











