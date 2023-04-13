import re

class Morph:
    def __init__(self, data):
        morph = data.replace("\t", ",").split(",")
        self.surface = morph[0]  # 表層形
        self.base = morph[7]  # 基本形
        self.pos = morph[1]  # 品詞
        self.pos1 = morph[2]

class Chunk:
    def __init__(self, morphs, dst, idx):
        self.morphs = morphs  # 中にはMorphクラスが入っている
        self.dst = dst  # 係り先文節インデックス番号
        self.srcs = []  # 係り元インデックス番号のリスト
        self.idx = int()
    

class Sentence:  # 係り先きを判別するには全文見る必要がある.よって文単位でのクラスを作る.
    def __init__(self, chunks):
        self.chunks = chunks  # SentenceクラスにはChunkクラスが入っている
        for i, chunk in enumerate(self.chunks):
            if chunk.dst != -1:  # 係り先が無い場合を排除する
                self.chunks[chunk.dst].srcs.append(i)  # 係り先を格納するdstの番号に対応するChunkをのリストに格納していく

def set_matrioshk(file_name):
    sentences = []
    chunks = []
    morphs = []
    idx = 0
    with open(file_name, "r") as data:
        for line in data:
            if line[0] == "*":  # IDを選択
                line = line.strip()
                if len(morphs) > 0:  # 既にmorphsに値が入っていればそれはchunkの終わりを示す
                    idx += 1
                    chunks.append(Chunk(morphs, dst, idx))  # Chunksがいっぱい入っているリストに格納
                    morphs = []  # ここでmorphsを初期化する.もししないと文が終わるまでmorphsに追加され続けるため、チャンクに分けられない.
                dst = int(re.search(r'(.*?)D', line.split()[2]).group(1))  # 係り先のインデックス番号からDを取り除いている

            elif line != "EOS\n":  # EOSが出ない場合
                morphs.append(Morph(line))  # 形態素が列挙されているのでChunkにするためにとりあえず格納

            else:
                if len(morphs) > 0:  # EOSが出た場合
                    chunks.append(Chunk(morphs, dst, idx))  # 文末が来たため、chunksに文の最後クラスを格納する
                    sentences.append(Sentence(chunks))  # 文末なので今までのchunksを格納する
                    morphs = []  # ここでも初期化
                    chunks = []  #次の文に備える
                    dst = None
                    idx = 0

    return sentences, chunks, morphs    