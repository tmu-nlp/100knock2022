'''
80. ID番号への変換
最も頻出する単語に1，2番目に頻出する単語に2…のように、学習データ中で2回以上出現する単語にID番号を付与.
与えられた単語列に対して，ID番号の列を返す関数を実装．
ただし，出現頻度が2回未満の単語のID番号はすべて0とする.'''
from collections import defaultdict
import pandas as pd
import string

# make ids for words according to frequency of occurrence from high to low

def make_ids4words(data):
    dic = defaultdict(int)
    for title in data.iloc[:,1]:
        table = str.maketrans(string.punctuation, " " * len(string.punctuation))
        for word in title.translate(table).split():
            if word not in ["0", "s"]:  # 51の前処理で全ての数字を0に置換、sが高い頻度であるのは全ての記号を空白に置換するため
                dic[word] += 1
    #頻度が高い順でソート
    dic = sorted(dic.items(), key=lambda x: x[1], reverse=True)
    #単語にidを付与
    word2id = {}
    for i, item in enumerate(dic):
        w, cnt = list(item)
        if cnt > 1:
            word2id[w] = i+1
    return word2id


# transform given words to ids
def get_id(sent, word2id, unk=0):
    res = []
    table = str.maketrans(string.punctuation, ' '*len(string.punctuation))
    for word in sent.translate(table).split():
        word = word.lower()
        res.append(word2id.get(word, unk))
    return res

if __name__ == '__main__':
    train_file = '../chapter06/train_re.txt'
    #train = pd.read_csv(train_file, header=None, names=['CATEGORY', 'TITLE'])
    train = pd.read_csv(train_file, header=None, sep='\t')
    word2id = make_ids4words(train)
    print(f'num of ids: {len(set(word2id.values()))}')
    print(f'---頻度上位20語―――')
    for w,cnt in list(word2id.items())[:20]:
        print(f'{w}:{cnt}')

    title = train.iloc[2, 1]      # 3行目の2列目
    print(f'title:{title}')
    print(f'ids:{get_id(title, word2id)}')
'''
num of ids: 7607
---頻度上位20語―――
to:1
in:2
the:3
of:4
for:5
on:6
as:7
update:8
us:9
and:10
a:11
with:12
at:13
is:14
after:15
new:16
stocks:17
says:18
up:19
from:20
title:joan rivers   joan rivers attacks lena dunham over  stay fat  message
ids:[1111, 1544, 1111, 1544, 1806, 766, 767, 26, 731, 1995, 1807]
'''