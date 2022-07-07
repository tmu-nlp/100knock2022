'''
50.  データの入手・整形
1. 情報源（publisher）が”Reuters”, “Huffington Post”, “Businessweek”, “Contactmusic.com”, “Daily Mail”の事例（記事）のみを抽出する;
2. 抽出された事例をランダムに並び替える;
3. train_data:80% -> train.txt, valid_data: 10%->valid.txt, test_data:10%->test.txt

tools:
document:https://scikit-learn.org/stable/modules/generated/sklearn.model_selection.train_test_split.html
pandas.DataFrame
model_selection.train_test_split(*arrays,test_size, train_size, random_state,shuffle, stratify) : split arrays or matrices into random train and test subsets.

random_state: pass an int for reproducible output across muliple function calls
stratify: if not None, data is split in an stratified fashion, using this as class labels
'''
import pandas as pd
from sklearn.model_selection import train_test_split



def get_data(file_path):
    #  列名：['ID', 'TITLE','URL','PUBLISHER','CATEGORY', 'STORY', 'HOSTNAME', 'TIMESTAMP']
    df = pd.read_csv(file_path, header=None, sep='\t',
                     names=['ID', 'TITLE', 'URL', 'PUBLISHER', 'CATEGORY', 'STORY', 'HOSTNAME', 'TIMESTAMP'])
    # 指定された情報源だけの事例を抽出
    # isinは条件抽出に使い、bool型を返す
    df = df.loc[df['PUBLISHER'].isin(['Reuters', 'Huffington Post', 'Businessweek', 'Contactmusic.com', 'Daily Mail']), ['CATEGORY', 'TITLE']]
    return df


if __name__ == '__main__':
    data_path = '../data/newsCorpora.csv'
    data = get_data(data_path)

    # データ分割
    train, val_test = train_test_split(data, test_size=0.2, shuffle=True, random_state=886, stratify=data['CATEGORY'])
    valid, test = train_test_split(val_test, test_size=0.5, shuffle=True, random_state=886, stratify=val_test['CATEGORY'])

    # データを保存の時、１行に１事例を書き出し，カテゴリ名と記事見出しのタブ区切り形式とする
    train.to_csv('./train.txt', columns=['CATEGORY', 'TITLE'], sep='\t', header=False, index=False)
    valid.to_csv('./valid.txt', columns=['CATEGORY', 'TITLE'], sep='\t', header=False, index=False)
    test.to_csv('./test.txt', columns=['CATEGORY', 'TITLE'], sep='\t', header=False, index=False)

    #value.counts(): ユニークな要素の値がindex、その出現個数がdataとなるpandas.Seriesを返す.
    print(train['CATEGORY'].value_counts())
    print(valid['CATEGORY'].value_counts())
    print(test['CATEGORY'].value_counts())


'''
b    4502
e    4223
t    1219
m     728
Name: CATEGORY, dtype: int64
b    562
e    528
t    153
m     91
Name: CATEGORY, dtype: int64
b    563
e    528
t    152
m     91
Name: CATEGORY, dtype: int64
'''
