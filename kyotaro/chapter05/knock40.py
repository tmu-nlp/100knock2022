"""
形態素を表すクラスMorphを実装せよ.このクラスは表層形（surface）,基本形（base）,品詞（pos）,品詞細分類1（pos1）
をメンバ変数に持つこととする.さらに,係り受け解析の結果（ai.ja.txt.parsed）を読み込み,
各文をMorphオブジェクトのリストとして表現し,冒頭の説明文の形態素列を表示せよ.
"""

import sys
from collections import defaultdict

class Morph:
    def __init__(self, data):
        morph = data.strip().replace("\t", ",").split(",")
        self.surface = morph[0]
        self.base = morph[7]
        self.pos = morph[1]
        self.pos1 = morph[2]

if __name__ == "__main__":
    sentences = []
    morphs = []
    with open(sys.argv[1], "r") as data:
        for line in data:
            if line[0] == "*":              # 1行目の文節番号などの情報は今回は無視
                continue
            elif line != "EOS\n":           # EOS以外（文末以外）の文を形態素の情報に1文ずつ入れていく
                morphs.append(Morph(line))
            else:                           # EOSが来たら文末が来たと判断して、それまでの1文をリストに追加していく
                sentences.append(morphs)
                morphs = []
    
    for m in sentences[2]:  #文章の番号を指定
        print(vars(m))  # 関数vars()を使ってクラスMorphを辞書型で返す
            