from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
import pandas as pd
import string
import re

def pre(text):
  table = str.maketrans(string.punctuation, ' '*len(string.punctuation))
  text = text.translate(table)  
  text = text.lower()  
  text = re.sub('[0-9]+', '0', text)  
  return text

df = pd.read_csv('./100knock2022/DUAN/chapter06/newsCorpora.csv', header=None, sep='\t', names=['ID', 'TITLE', 'URL', 'PUBLISHER', 'CATEGORY', 'STORY', 'HOSTNAME', 'TIMESTAMP'])
df = df.loc[df['PUBLISHER'].isin(['Reuters', 'Huffington Post', 'Businessweek', 'Contactmusic.com', 'Daily Mail']), ['TITLE', 'CATEGORY']]
train, valid_test = train_test_split(df, test_size=0.2, shuffle=True, random_state=123, stratify=df['CATEGORY'])
valid, test = train_test_split(valid_test, test_size=0.5, shuffle=True, random_state=123, stratify=valid_test['CATEGORY'])
train.to_csv('./100knock2022/DUAN/chapter06/train.txt', sep='\t', index=False)
valid.to_csv('./100knock2022/DUAN/chapter06/valid.txt', sep='\t', index=False)
test.to_csv('./100knock2022/DUAN/chapter06/test.txt', sep='\t', index=False)

df = pd.concat([train, valid, test], axis=0)
df.reset_index(drop=True, inplace=True) 
df['TITLE'] = df['TITLE'].map(lambda x: pre(x))

train_valid = df[:len(train) + len(valid)]
test = df[len(train) + len(valid):]
vec_tfidf = TfidfVectorizer(min_df=10, ngram_range=(1, 2)) 

X_train_valid = vec_tfidf.fit_transform(train_valid['TITLE']) 
X_test = vec_tfidf.transform(test['TITLE'])
X_train_valid = pd.DataFrame(X_train_valid.toarray(), columns=vec_tfidf.get_feature_names_out())
X_test = pd.DataFrame(X_test.toarray(), columns=vec_tfidf.get_feature_names_out())

X_train = X_train_valid[:len(train)]
X_valid = X_train_valid[len(train):]

X_train.to_csv('./100knock2022/DUAN/chapter06/train.feature.txt', sep='\t', index=False)
X_valid.to_csv('./100knock2022/DUAN/chapter06/valid.feature.txt', sep='\t', index=False)
X_test.to_csv('./100knock2022/DUAN/chapter06/test.feature.txt', sep='\t', index=False)

lg = LogisticRegression(random_state=123, max_iter=10000)
lg.fit(X_train, train['CATEGORY'])