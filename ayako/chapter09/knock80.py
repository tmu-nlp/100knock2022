#問題51で構築した学習データ中の単語にユニークなID番号を付与したい.
#学習データ中で最も頻出する単語に1,2番目に頻出する単語に2,……といった方法で, 学習データ中で2回以上出現する単語にID番号を付与せよ.
#そして,与えられた単語列に対して,ID番号の列を返す関数を実装せよ.ただし,出現頻度が2回未満の単語のID番号はすべて0とせよ.

from collections import defaultdict
import pandas as pd

def count_word(data):
    """単語の頻度をカウントして頻度順にソート"""
    word_cnt = defaultdict(lambda:0)
    for text in data:
        for word in text.strip().split():
            word_cnt[word] += 1
    word_cnt = sorted(word_cnt.items(), key=lambda x:x[1], reverse=True)
    #("word","cnt"),...
    return word_cnt

def word2id(word_cnt):
    """単語ID辞書を生成"""
    word2id_dic = {}
    for i, (word, cnt) in enumerate(word_cnt):
        if cnt < 2:#出現頻度2回以上の単語のみ辞書に追加
            continue
        word2id_dic[word] = i+1#頻度1番目に1，2番目に2,...
    return word2id_dic

def tokenizer(text, word2id_dic, unk=0):
    """タイトルをtokenizeしてid列を返す"""
    ids = []
    for word in text.strip().split():
        ids.append(word2id_dic.get(word, unk))
    return ids

if __name__ == "__main__":
    #knock50で作成したデータを読み込む
    train = pd.read_csv("../chapter06/train.txt", header=None, sep="\t")
    valid = pd.read_csv("../chapter06/valid.txt", header=None, sep="\t")
    test = pd.read_csv("../chapter06/test.txt", header=None, sep="\t")

    colums_name = ["TITLE", "CATEGORY"]
    train.columns = colums_name
    valid.columns = colums_name
    test.columns = colums_name

    word_cnt = count_word(train["TITLE"])
    word2id_dic = word2id(word_cnt)

    #結果を表示
    text = train.iloc[0,train.columns.get_loc('TITLE')]
    print(f"テキスト: {text}")
    print(f"ID列: {tokenizer(text, word2id_dic)}")

"""
テキスト: Agnellis keen to support Fiat Chrysler going forward: Elkann
ID列: [0, 4923, 1, 983, 1173, 1066, 1174, 0, 0]
"""
