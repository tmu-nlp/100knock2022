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
            if not chunk is  None:
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
                if not chunk is  None:
                    sentence_chunk.append(chunk)
                chunk = Chunk(int(line.split()[2][:-1]))
            else:
                chunk.add_morph(Morph(line.rstrip()))

with open('./100knock2022/DUAN/chapter05/knock46.txt', mode='w') as f:       
    for chunks in sentences_chunk:
        for chunk in chunks:
            if len(chunk.srcs)== 0:
                continue
            src_chunks = [chunks[i] for i in chunk.srcs]
            if '動詞' in [morph.pos for morph in chunk.morphs]:
                verb = [morph.base for morph in chunk.morphs if morph.pos == '動詞'][0]
                pp_term = []
                for chunk in src_chunks:
                    for morph in chunk.morphs[::-1]:
                        if morph.pos == '助詞':
                            term = "".join([morph.surface for morph in chunk.morphs if morph.pos != "記号"])
                            pp_term.append((morph.base, term))
                            break
                pp_term = sorted(pp_term, key=lambda x: x[0])
                if len(pp_term) == 0: 
                    continue
                pp, term = map(list, zip(*pp_term))
                f.write(verb+'\t'+''.join(pp)+'\t'+' '.join(term)+'\n')
                