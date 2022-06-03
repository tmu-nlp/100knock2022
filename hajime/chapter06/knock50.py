# ダブルクオテーションをシングルクオテーションに変更
# 読み込みのバグ回避
# sed -e 's/"/'\''/g' ./newsCorpora.csv > ./newsCorpora_re.csv

import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split


extract_list = ['Reuters', 'Huffington Post',
                'Businessweek', 'Contactmusic.com', 'Daily Mail']
header_name = ['ID', 'TITLE', 'URL', 'PUBLISHER', 'CATEGORY', 'STORY', 'HOSTNAME', 'TIMESTAMP'
               ]

# pandasで読み込み
# headerはなし．分割はタブ．headerの名前はheader_name．
df = pd.read_csv('./newsCorpora_re.csv', header=None,
                 sep='\t', names=header_name)

# 抽出
# dfの'PUBLISHER'の要素に対しextract_listの要素である場合は，'TITLE'と'CATEGORY'を抽出
df_ex = df.loc[df['PUBLISHER'].isin(extract_list), ['TITLE', 'CATEGORY']]

data_train, data_other = train_test_split(
    df_ex, test_size=0.2, random_state=20010101)
data_valid, data_test = train_test_split(
    data_other, test_size=0.5, random_state=20010101)

data_train.to_csv("train.txt", sep="\t", index=False, header=False)
data_valid.to_csv("valid.txt", sep="\t", index=False, header=False)
data_test.to_csv("test.txt", sep="\t", index=False, header=False)

print("train_data")
print(data_train['CATEGORY'].value_counts())
print("valid_data")
print(data_valid['CATEGORY'].value_counts())
print("test_data")
print(data_test['CATEGORY'].value_counts())
