from collections import defaultdict
import string
import pandas as pd
from funcs import picklize

def tokenizer(text, word2id, unleg=0):
    """入力テキストをスペースで分割しID列に変換(辞書になければunlegで指定した数字を設定)"""
    table = str.maketrans(string.punctuation, " " * len(string.punctuation))
    res = []
    for word in text.translate(table).split():
        res.append(word2id.get(word, unleg))

    return res


train = pd.read_csv("train.txt", sep="\t")
valid = pd.read_csv("valid.txt", sep="\t")
test = pd.read_csv("test.txt", sep="\t")

# 単語の頻度集計
d = defaultdict(int)
table = str.maketrans(
    string.punctuation, " " * len(string.punctuation)
)  # 記号をスペースに置換するテーブル
for text in train["TITLE"]:
    for word in text.translate(table).split():
        d[word] += 1
d = sorted(d.items(), key=lambda x: x[1], reverse=True)  # 出現回数の降順でソート


word2id = {}  # 単語IDを保持する辞書
for i, (word, cnt) in enumerate(d):
    if cnt > 1:
        word2id[word] = i + 1

# 確認用
text = train.iloc[0, train.columns.get_loc("TITLE")]  # 行番号0のTITLE
print(f"テキスト: {text}")
print(f"ID列: {tokenizer(text, word2id)}")

to_pickle = {"word2id": word2id}
picklize(to_pickle)