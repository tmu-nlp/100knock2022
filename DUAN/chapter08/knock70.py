import string
import torch
import pandas as pd
from gensim.models import KeyedVectors
from sklearn.model_selection import train_test_split

df = pd.read_csv('./100knock2022/DUAN/chapter06/newsCorpora.csv', header=None, sep='\t', names=['ID', 'TITLE', 'URL', 'PUBLISHER', 'CATEGORY', 'STORY', 'HOSTNAME', 'TIMESTAMP'])
df = df.loc[df['PUBLISHER'].isin(['Reuters', 'Huffington Post', 'Businessweek', 'Contactmusic.com', 'Daily Mail']), ['TITLE', 'CATEGORY']]
train, valid_test = train_test_split(df, test_size=0.2, shuffle=True, random_state=123, stratify=df['CATEGORY'])
valid, test = train_test_split(valid_test, test_size=0.5, shuffle=True, random_state=123, stratify=valid_test['CATEGORY'])
model = KeyedVectors.load_word2vec_format('./100knock2022/DUAN/chapter08/GoogleNews-vectors-negative300.bin', binary=True)

def transform_w2v(text):
    table = str.maketrans(string.punctuation, ' '*len(string.punctuation))
    words = text.translate(table).split()  
    vec = [model[word] for word in words if word in model]  
    return torch.tensor(sum(vec) / len(vec))

X_train = torch.stack([transform_w2v(text) for text in train['TITLE']])
X_valid = torch.stack([transform_w2v(text) for text in valid['TITLE']])
X_test = torch.stack([transform_w2v(text) for text in test['TITLE']])
category_dict = {'b': 0, 't': 1, 'e':2, 'm':3}
y_train = torch.tensor(train['CATEGORY'].map(lambda x: category_dict[x]).values)
y_valid = torch.tensor(valid['CATEGORY'].map(lambda x: category_dict[x]).values)
y_test = torch.tensor(test['CATEGORY'].map(lambda x: category_dict[x]).values)

torch.save(X_train, './100knock2022/DUAN/chapter08/X_train.pt')
torch.save(X_valid, './100knock2022/DUAN/chapter08/X_valid.pt')
torch.save(X_test, './100knock2022/DUAN/chapter08/X_test.pt')
torch.save(y_train, './100knock2022/DUAN/chapter08/y_train.pt')
torch.save(y_valid, './100knock2022/DUAN/chapter08/y_valid.pt')
torch.save(y_test, './100knock2022/DUAN/chapter08/y_test.pt')