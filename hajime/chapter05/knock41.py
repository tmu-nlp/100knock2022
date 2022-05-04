# knock41
# 40に加えて，文節を表すクラスChunkを実装せよ．
# このクラスは形態素（Morphオブジェクト）のリスト（morphs），係り先文節インデックス番号（dst），
# 係り元文節インデックス番号のリスト（srcs）をメンバ変数に持つこととする．
# さらに，入力テキストの係り受け解析結果を読み込み，１文をChunkオブジェクトのリストとして表現し，
# 冒頭の説明文の文節の文字列と係り先を表示せよ．
# 本章の残りの問題では，ここで作ったプログラムを活用せよ．

# class Chunk
# morphs : 形態素(Morphオブジェクトのリスト)
# dst : 係り先文節インデックス番号
# srcs : 係り元文節インデックス番号

# CaBoCha
# https://taku910.github.io/cabocha/

# * 27 34D 3/3 1.275360
# 文節の区切り情報(*) 文節番号 係り受け先+D 主語/機能語の位置 係り関係のスコア
# それぞれ空白区切りで出力

# ひとまずの目標 -> morphsとdstの情報を書き込む
# できたらsrcsを頑張る

class Morph():
    def __init__(self, morph):
        surface, attr = morph.split('\t')
        attr_list = attr.split(',')
        self.surface = surface
        self.base = attr_list[6]
        self.pos = attr_list[0]
        self.pos1 = attr_list[1]


class Chunk():
    def __init__(self, morphs, dst):
        self.morphs = morphs  # 形態素のリスト
        self.dst = dst  # 係り先文節インデックス番号
        self.srcs = []  # 係り元文節インデックス番号のリスト


class Sentence():
    def __init__(self, chunks):
        self.chunks = chunks  # Chunk型のリスト ここに一文の全てのchunkが揃っている
        for i, chunk in enumerate(self.chunks):  # 番号が欲しいためenumerateでfor文
            if chunk.dst != -1:  # 係り受け先が存在する場合
                self.chunks[chunk.dst].srcs.append(i)
                # (self.chunks[chunk.dst].srcs)にiを追加
                # chunks[chunk.dst]は係り受け先のindex
                # .srcsは係り元文節インデックス番号のリスト


sentences = []
morphs = []
chunks = []

with open("ai.ja.txt.parsed", "r") as f:
    for line in f:
        if line[0] == '*':
            if len(morphs) > 0:
                chunks.append(Chunk(morphs, dst))  # chunkの区切り目になるのでappend
                morphs = []  # 初期化
            dst = int(line.split(' ')[2].rstrip('D'))  # (num)DのDを除去してint化
        elif line == "EOS\n":
            if len(morphs) > 0:  # morphの中身がある場合
                chunks.append(Chunk(morphs, dst))  # chunksにchunkオブジェクトを追加
                sentences.append(Sentence(chunks))  # sentenceの区切りめになるのでappend
            morphs = []
            chunks = []
            dst = None
        else:
            morphs.append(Morph(line))  # morphを追加

for chunk in sentences[1].chunks:
    print([morph.surface for morph in chunk.morphs], chunk.dst, chunk.srcs)
