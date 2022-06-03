import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split

df = pd.read_csv('newsCorpora.csv', sep = '\t', header=None, names = ['ID', 'TITLE', 'URL', 'PUBLISHER', 'CATEGORY', 'STORY', 'HOSTNAME', 'TIMESTAMP'])

#要領2 出版元の抽出
flag = df['PUBLISHER'].isin(['Reuters', 'Huffington Post', 'Businessweek', 'Contactmusic.com', 'Daily Mail'])
df = df[flag]
    #まとめて書いた場合が以下の通り
    #df = df.loc[df['PUBLISHER'].isin(['Reuters', 'Huffington Post', 'Businessweek', 'Contactmusic.com', 'Daily Mail'])]
    #len(df) #13340

#要領3 データのシャッフル
df = df.sample(frac=1, random_state=0)
    #sampleメソッド...dfをランダムサンプリング
    #frac引数...抽出する行・列の割合。1だと100%指定、再現性確保のためシード値は0に設定

#要領4 データの分割
train_data, other_data = train_test_split(df, test_size=0.2)
valid_data, test_data = train_test_split(other_data, test_size=0.5)

#テキストに書き出し
train_data.to_csv('./train.txt', sep = '\t', index = False)
valid_data.to_csv('./valid.txt', sep = '\t', index = False)
test_data.to_csv('./test.txt', sep = '\t', index = False)

#各カテゴリの事例数の確認
print('学習データ')
print(train_data['CATEGORY'].value_counts())
print('\n検証データ')
print(valid_data['CATEGORY'].value_counts())
print('\nテストデータ')
print(test_data['CATEGORY'].value_counts())
