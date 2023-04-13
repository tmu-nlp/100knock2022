# knock41
# 係り受け解析結果の読み込み（文節・係り受け）
# 40に加えて，文節を表すクラスChunkを実装せよ．このクラスは形態素（Morphオブジェクト）のリスト（morphs），
# 係り先文節インデックス番号（dst），係り元文節インデックス番号のリスト（srcs）をメンバ変数に持つこととする．
# さらに，入力テキストの係り受け解析結果を読み込み，１文をChunkオブジェクトのリストとして表現し，冒頭の説明文の文節の文字列と係り先を表示せよ．
# 本章の残りの問題では，ここで作ったプログラムを活用せよ．
import re

class Morph:
    def __init__(self, surface, base, pos, pos1):
        self.surface = surface
        self.base = base
        self.pos = pos
        self.pos1 = pos1
    def __str__(self):
        return f"表層形:{self.surface}\t基本形:{self.base}\t品詞:{self.pos}\t品詞細分類1:{self.pos1}"

class Chunk:
    def __init__(self):
        self.morphs = []
        self.index = -1 # 文節番号
        self.dst = -1 # 係り先
        self.srcs = [] # 係り元
    def __str__(self):
        surface = ""
        for morph in self.morphs:
            surface += morph.surface
        return f"surface:{surface}\tdst:{self.dst}\tsrcs{self.srcs}"


def parse_morpheme(line):
    line = line.split("\t")  # タブ区切りでsurfaceとその他を分割
    new_line = line[1].split(",")
    return Morph(line[0], new_line[6], new_line[0], new_line[1])

def parse_chunk(fname):
    text = []  # 文書全体のリスト
    with open(fname, "r") as input_file:
        sentence = []  # 一文の文節まとめるリスト
        morphs = []
        chunk = Chunk()  # 文節初期化

        for line in input_file:
            if line[0] == "*":  # 文節の初め
                line = line.split()
                if line[1] != "0":  # 前のループで作ったchunkをリストに追加
                    sentence.append(chunk)
                chunk = Chunk()  # chunkを初期化
                chunk.index = int(line[1])
                num = re.findall(r'^(.*?)D', line[2])  # リストで返る
                line[2] = num[0]
                chunk.dst = int(line[2])

            elif line == "EOS\n":  # 文の終わりの時
                if len(morphs) > 0:  # 形態素入ってたら文節完成
                    sentence.append(chunk)
                    text.append(sentence)  # 文書全体のリストに一文分追加
                sentence = []  # 文節まとめリストを初期化
                morphs = []

            else:
                morph = parse_morpheme(line)
                morphs.append(morph)
                chunk.morphs.append(morph)

    for sentence in text:
        for chunk in sentence:
            dst = chunk.dst
            if dst == -1:  # -1の時は係先なし
                continue
            sentence[dst].srcs.append(chunk.index)
    return text

if __name__ == "__main__":
    fname = "ai.ja.txt.parsed"
    text = parse_chunk(fname)
    for sentence in text[1:2]:  # 冒頭の説明文だけ
        for chunk in sentence:
            print(chunk)