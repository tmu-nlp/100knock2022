'''
41.係り受け解析結果の読み込み(文節・係り受け)
文節を表すクラスChunkを実装
このクラスは形態素（Morphオブジェクト）のリスト（morphs），係り先文節インデックス番号（dst），係り元文節インデックス番号のリスト（srcs）をメンバ変数に持つこと
入力テキストの係り受け解析結果を読み込み，１文をChunkオブジェクトのリストとして表現し，冒頭の説明文の文節の文字列と係り先を表示
本問で作ったプログラムは以降も使う

*をはじめとする行で文節chunkを定義して、morphsからなる。chunks==sentenceはchunkからなる。
文末（EOS）が来た場合、chunk objectをchunksに追加し、1文の文節をchunksにまとめる。
係り先の文節番号dst：parsed file中でDの前の数字
係り元の文節番号srcs:
sentencesはSentence classを持つ文リストあり、sentence classには文節リスト(chunks)を持つ。
chunksはChunk classを持つ文節リストであり、chunk classには文節ごとの形態素リスト、係り先文章番号、係り元文章番号、文章番号を持つ
Morph classには40のように各行の形態素を持つ
'''
from knock40 import Morph, get_morphs


class Chunk:
    def __init__(self, morphs, dst, chunk_id):
        self.morphs = morphs      # Morph object のリスト
        self.dst = dst
        self.srcs = []            # index of chunk in chunks
        self.chunk_id = chunk_id


class Sentence:
    def __init__(self, chunks):
        self.chunks = chunks       # Chunk objectのリスト
        for i, chunk in enumerate(self.chunks):
            if chunk.dst not in [None, -1]:
                self.chunks[chunk.dst].srcs.append(i)



def get_chunks(file):

    sentences = []    # 文リスト
    chunks = []       # 節リスト
    morphs = []       # 形態素リスト
    chunk_id = 0      # 文節番号

    with open(file, 'r', encoding='utf-8') as f:
        for line in f:
            if line[0] == '*':  # 文節区切り、係り先情報入る: * 0 -1D 1/1 0.000000
                if len(morphs) > 0:
                    chunks.append(Chunk(morphs, dst, chunk_id))
                    chunk_id += 1
                    morphs = []    # morphs に入れない
                dst = int(line.split()[2].replace('D',''))   # 係り先文節番号を取得
            elif line != 'EOS\n':
                morphs.append(Morph(line))    # morphsをリストに保存
            else:
                if len(morphs) > 0:                # EOS行の情報は入れない
                    chunks.append(Chunk(morphs, dst, chunk_id))    # 文節のリストに追加

                    sentences.append(Sentence(chunks))             # 文のリストに追加

                morphs = []
                chunks = []
                dst = None
                chunk_id = 0

    return sentences


# 出力結果を確認
if __name__ == '__main__':
    file_path = '../data/ai.ja.txt.parsed'
    sentences = get_chunks(file_path)

    for chunk in sentences[1].chunks:
        chunk_str = ''.join([morph.surface for morph in chunk.morphs])
        print(f'文節の文字列:{chunk_str}\t係り先の文節番号:{chunk.dst}')








