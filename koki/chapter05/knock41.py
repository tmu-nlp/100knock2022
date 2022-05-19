#係り受け解析結果の出力形式は以下の通り
# * \s 文節番号 \s 係り先の文節番号 \s 主辞/機能語の位置 \s 係り関係のスコア
import re

separator = re.compile('\t|,') #globalかクラス内のlocalのどっちで定義すべきか？

class Morph():
    def __init__(self, morph):
        values = separator.split(morph)# 「\t」 or「,」区切りで分割
        self.surface = values[0] #表層形
        self.base = values[7] #基本形
        self.pos = values[1] #品詞
        self.pos1 = values[2] #品詞細分類1

class Chunk(): #文節を管理
    def __init__(self, morphs, dst):
        self.morphs = morphs #Morphオブジェクトのリスト
        self.dst = dst #係り先番号
        self.srcs = [] #係り元番号, 複数存在する可能性があるためリストで定義

class Sentence(): #1文を管理
    def __init__(self, chunks):
        self.chunks = chunks #Chunk型配列
        for i, chunk in enumerate(self.chunks): #文節のインデックス番号を文節(chunk)リストにより数える
            if chunk.dst != -1: #係り先が存在する場合
                self.chunks[chunk.dst].srcs.append(i) #係り元番号を追加

morphs = [] #形態素ごと解析結果のオブジェクト配列
sentences = [] #1フレーズごとに解析結果を管理、Setence型オブジェクト配列
chunks = [] #文節を管理

with open('ai.ja.txt.parsed') as f:
    for line in f:
        if line[0] == '*': #係り受け解析結果は無視
            if len(morphs) > 0:
                chunks.append(Chunk(morphs, dst))
                morphs = []
            dst = int(line.split(' ')[2].strip('D'))#掛かり先情報「2D」などのDは不要なためDを削除

        elif line == 'EOS\n': #EOSも無視(\nがないとだめだった)
            if len(morphs) > 0: #連続してEOSが続く場合など、空文字がsentencesに格納されることを防ぐ
                chunks.append(Chunk(morphs, dst))
                sentences.append(Sentence(chunks))
                morphs = []
            chunks = []
            dst = None
            
        else: 
            morph_result = Morph(line) #インスタンス化
            morphs.append(morph_result)

if __name__ == '__main__':
    for chunk in sentences[1].chunks:
        print([morph.surface for morph in chunk.morphs], chunk.dst, chunk.srcs)
        #[形態素リスト],  掛かり先インデックス,  係り元インデックス 
