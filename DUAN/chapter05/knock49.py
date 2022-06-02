import copy
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
    
with open('./100knock2022/DUAN/chapter05/knock48.txt', mode='w') as f:   
    dict_pathes = dict()
    lines = []
    for j, chunks in enumerate(sentences_chunk):
        pathes = []
        path = []
        for i, chunk in enumerate(chunks):
            if '名詞' in [morph.pos for morph in chunk.morphs]:
                line = [''.join([morph.surface for morph in chunk.morphs if morph.pos != '記号'])]
                path = [i]
                now_chunk = chunk
                while now_chunk.dst != -1:
                    if '句点' in [morph.pos1 for morph in now_chunk.morphs]:
                        break
                    path.append(now_chunk.dst)
                    now_chunk = chunks[now_chunk.dst]
                    line.append(''.join([morph.surface for morph in now_chunk.morphs if morph.pos != '記号']))
                if len(line) > 1:
                    f.write(' -> '.join(line) + '\n')
                    lines.append(line)
                    pathes.append(path)
        dict_pathes[j] = pathes

with open('./100knock2022/DUAN/chapter05/knock49.txt', mode='w') as f:
    for i, pathes in dict_pathes.items():
        for j, path_X in enumerate(pathes[:-1]):
            for path_Y in pathes[j+1:]:
                s_X = set(path_X)
                s_Y = set(path_Y)
                if s_X >= s_Y:
                    idx = sorted(list(s_X - s_Y) + [path_Y[0]])
                    chunks = copy.deepcopy(sentences_chunk[i])
                    chunks = [chunks[k] for k in range(len(chunks)) if k in idx]
                    if not '名詞' in chunks[0].morphs and '名詞' in chunks[0].morphs:
                        continue
                    for n, morph in enumerate(chunks[0].morphs):
                        if morph.pos == '記号':
                            continue
                        if morph.pos == '名詞' or morph.pos == '接頭詞':
                            chunks[0].morphs[n].surface = 'X'
                        else:
                            break
                    for n, morph in enumerate(chunks[-1].morphs):
                        if morph.pos == '記号':
                            continue
                        if morph.pos == '名詞' or morph.pos == '接頭詞':
                            chunks[-1].morphs[n].surface = 'Y'
                        else:
                            break
                    s = ' -> '.join([''.join([morph.surface for morph in chunk.morphs if morph.pos != '記号']) for chunk in chunks])
                    s = re.sub('X+', 'X', s)
                    s = re.sub('Y+', 'Y', s)
                    print(s, file=f)
                elif len(s_X - s_Y) == 1:
                    px = copy.deepcopy(path_X)
                    py = copy.deepcopy(path_Y)
                    px.pop()
                    py.pop()
                    chunks = copy.deepcopy(sentences_chunk[i])
                    chunks_X = [chunks[k] for k in range(len(chunks)) if k in px]
                    chunks_Y = [chunks[k] for k in range(len(chunks)) if k in py]
                    last_chunk = chunks[path_X[-1]]
                    for n, morph in enumerate(chunks_X[0].morphs):
                        if morph.pos == '記号':
                            continue
                        if morph.pos == '名詞' or morph.pos == '接頭詞':
                            chunks_X[0].morphs[n].surface = 'X'
                        else:
                            break
                    for n, morph in enumerate(chunks_Y[0].morphs):
                        if morph.pos == '記号':
                            continue
                        if morph.pos == '名詞' or morph.pos == '接頭詞':
                            chunks_Y[0].morphs[n].surface = 'Y'
                        else:
                            break
                    s_x = ' -> '.join([''.join([morph.surface for morph in chunk.morphs if morph.pos != '記号']) for chunk in chunks_X])
                    s_y = ' -> '.join([''.join([morph.surface for morph in chunk.morphs if morph.pos != '記号']) for chunk in chunks_Y])
                    s_l = ''.join([morph.surface for morph in last_chunk.morphs if morph.pos != '記号'])
                    s = s_x + ' | ' + s_y + ' | ' + s_l
                    s = re.sub('X+', 'X', s)
                    s = re.sub('Y+', 'Y', s)
                    print(s, file=f)