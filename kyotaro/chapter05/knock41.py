"""
40に加えて,文節を表すクラスChunkを実装せよ.このクラスは形態素（Morphオブジェクト）のリスト（morphs）,
係り先文節インデックス番号（dst）,係り元文節インデックス番号のリスト（srcs）をメンバ変数に持つこととする.
さらに,入力テキストの係り受け解析結果を読み込み,1文をChunkオブジェクトのリストとして表現し,
冒頭の説明文の文節の文字列と係り先を表示せよ.本章の残りの問題では,ここで作ったプログラムを活用せよ.
"""

import sys
import re
from collections import defaultdict

class Morph:
    def __init__(self, data):
        morph = data.replace("\t", ",").split(",")
        self.surface = morph[0]
        self.base = morph[7]
        self.pos = morph[1]
        self.pos1 = morph[2]

class Chunk:
    def __init__(self, morphs, dst):
        self.morphs = morphs
        self.dst = dst
        self.srcs = []
    

class Sentence:  # 係り先きを判別するには全文見る必要がある.よって文単位でのクラスを作る.
    def __init__(self, chunks):
        self.chunks = chunks
        for i, chunk in enumerate(self.chunks):
            if chunk.dst != -1:  # 係り先きが無い場合を排除する
                self.chunks[chunk.dst].srcs.append(i)  # 係り先きを格納するChunkのリストに格納していく



sentences = []
chunks = []
morphs = []

with open(sys.argv[1], "r") as data:
    for line in data:
        if line[0] == "*":
            line = line.strip()
            if len(morphs) > 0:
                chunks.append(Chunk(morphs, dst))
                morphs = []  # ここでmorphsを初期化する.もししないと文が終わるまでmorphsに追加され続けるため、チャンクに分けられない.
            dst = int(re.search(r'(.*?)D', line.split()[2]).group(1))  # 係り先のインデックス番号からDを取り除いている

        elif line != "EOS\n":  # EOSが出ない場合
            morphs.append(Morph(line))

        else:
            if len(morphs) > 0:
                chunks.append(Chunk(morphs, dst))  # 文末が来たため、chunksに文の最後クラスを格納する
                sentences.append(Sentence(chunks))
                morphs = []  # ここでも初期化
                chunks = []  #次の文に備える
    
for chunk in sentences[20].chunks:
    print([morph.surface for morph in chunk.morphs], chunk.dst, chunk.srcs)
