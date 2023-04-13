"""
形態素を表すクラスMorphを実装せよ. このクラスは表層形（surface）, 基本形（base）, 品詞（pos）, 
品詞細分類1（pos1）をメンバ変数に持つこととする. さらに, 係り受け解析の結果（ai.ja.txt.parsed）を読み込み, 
各文をMorphオブジェクトのリストとして表現し, 冒頭の説明文の形態素列を表示せよ.
"""

import sys

class Morph:
    def __init__(self, data):
        morph = data.replace("\t", ",").split(",")
        self.surface = morph[0]  # 表層形
        self.base = morph[7]  # 基本形
        self.pos = morph[1]  # 品詞
        self.pos1 = morph[2]

morphs = []
sentences = []

with open(sys.argv[1], "r") as data:
    for line in data:
        if line[0] == "*":  # IDを選択
            continue

        elif line != "EOS\n":  # EOSが出ない場合
            morphs.append(Morph(line))  # 形態素が列挙されているのでChunkにするためにとりあえず格納

        else:
            if len(morphs) > 0:  # EOSが出た場合
                sentences.append(morphs)
                morphs = []  # ここでも初期化

for morph in sentences[2]:
    print(vars(morph))