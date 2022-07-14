# 47. 機能動詞構文のマイニングPermalink
# 係り先文節インデックス番号（dst） 係り元文節インデックス番号のリスト（srcs）

import re

class Morph:
    def __init__(self, morph):
        morph = re.split('[\t,]', morph)
        if len(morph) >= 8:
            self.surface = morph[0]
            self.base = morph[7]
            self.pos = morph[1]
            self.pos1 = morph[2]

class Chunk:
    def __init__(self, dst):
        self.dst = dst
        self.srcs = []
        self.morphs = []
        
    def add_src(self, src: int):
        self.srcs.append(src)
        
    def add_morph(self, morph: Morph):
        self.morphs.append(morph)
        
with open('./100knock2022/DUAN/chapter05/ai.ja.txt.parsed') as f_parsed:
    sentences_chunk = []
    sentence_chunk = []
    chunk = None
    for line in f_parsed:
        if line.startswith('EOS'):
            if not chunk is None:
                sentence_chunk.append(chunk)
                chunk = None
            if len(sentence_chunk) != 0:
                for i, c in enumerate(sentence_chunk):
                    if c.dst != -1:
                        sentence_chunk[c.dst].add_src(i)
                sentences_chunk.append(sentence_chunk)
            sentence_chunk = []
        else:
            if line.startswith('*'):
                if not chunk is None:
                    sentence_chunk.append(chunk)
                chunk = Chunk(int(line.split()[2][:-1]))
            else:
                chunk.add_morph(Morph(line.rstrip()))

with open('./100knock2022/DUAN/chapter05/knock47.txt', mode='w') as f:
    for chunks in sentences_chunk:
        for chunk in chunks:
            # 係り元のチェック
            if len(chunk.srcs) == 0:
                continue
            src_chunks = [chunks[i] for i in chunk.srcs]
            if '動詞' in [morph.pos for morph in chunk.morphs]: 
                    # 動詞の基本形を取得する
                    verb = [morph.base for morph in chunk.morphs if morph.pos == '動詞'][0]
                    pp_term = [] # 助詞と項
                    sahen_term = None # サ変動詞接続名詞
                    #「サ変接続名詞+を（助詞）」で構成された文節が、動詞に係っていれば、「サ変接続名詞+を+動詞の基本形」の形で保存する
                    for chunk in src_chunks[::-1]:
                        for i, morph in enumerate(chunk.morphs[::-1]):
                            # 述語に係る助詞を取得していく
                            if morph.pos == '助詞':
                                term = ''.join([morph.surface for morph in chunk.morphs if morph.pos != '記号'])
                                pp_term.append((morph.base, term))
                                if morph.base == 'を' and chunk.morphs[-(i+2)].pos1 == 'サ変接続' and sahen_term is None:
                                    sahen_term = chunk.morphs[-(i+2)].surface + 'を'
                                    pp_term.pop()
                                break
                    if sahen_term is None:
                        continue
                    # 辞書順に並べる
                    pp_term = sorted(pp_term, key=lambda x: x[0])
                    if len(pp_term) == 0: 
                        continue
                    pp, term = map(list, zip(*pp_term))
                    # 述語と述語に係っている助詞、項をタブ区切りでファイルに書き込む
                    f.write(sahen_term+verb+'\t'+' '.join(pp)+'\t'+' '.join(term)+'\n')
