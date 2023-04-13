import pandas as pd
from sklearn.model_selection import train_test_split
from collections import defaultdict
import string


def tokenizer(text, dic):
    cnt = 0
    table = str.maketrans(string.punctuation, ' '*len(string.punctuation))
    return [dic.get(word, cnt) for word in text.translate(table).split()]


# csv to DataFrame
df = pd.read_csv('newsCorpora.csv', sep='\t', header=None, names=[
                 'ID', 'TITLE', 'URL', 'PUBLISHER', 'CATEGORY', 'STORY', 'HOSTNAME', 'TIMESTAMP'])

# 出版元の抽出
flag = df['PUBLISHER'].isin(
    ['Reuters', 'Huffington Post', 'Businessweek', 'Contactmusic.com', 'Daily Mail'])
df = df[flag]

# データの分割 train: valid: test = 0.4: 0.4: 0.2
train_data, other_data = train_test_split(df, test_size=0.2)
valid_data, test_data = train_test_split(other_data, test_size=0.5)

# 単語の頻度カウント
d = defaultdict(int)
table = str.maketrans(string.punctuation, ' '*len(string.punctuation))
for text in train_data['TITLE']:
    for word in text.translate(table).split():
        d[word] += 1
d = sorted(d.items(), key=lambda x: x[1], reverse=True)

# 単語ID辞書の作成
word_id = defaultdict(int)
for i, (word, cnt) in enumerate(d):
    if cnt > 1:
        word_id[word] = i+1

category_dict = {'b': 0, 't': 1, 'e': 2, 'm': 3}

y_train = train_data['CATEGORY'].map(lambda x: category_dict[x]).values
train_data = train_data['TITLE']
train_data = pd.Series(train_data, index=None)
train_data = train_data.reset_index(drop=True)

y_valid = valid_data['CATEGORY'].map(lambda x: category_dict[x]).values
valid_data = valid_data['TITLE']
valid_data = pd.Series(valid_data, index=None)
valid_data = valid_data.reset_index(drop=True)

y_test = test_data['CATEGORY'].map(lambda x: category_dict[x]).values
test_data = test_data['TITLE']
test_data = pd.Series(test_data, index=None)
test_data = test_data.reset_index(drop=True)
