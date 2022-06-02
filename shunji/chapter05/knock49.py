from itertools import combinations
import re
from knock41 import sentences


for sentence in sentences:
    nouns = []  # 名詞を含むチャンクのインデックスを格納するリスト
    for i, chunk in enumerate(sentence.chunks):
        if '名詞' in [morph.pos for morph in chunk.morphs]:  # chunkに名詞の形態素が入っているか
            nouns.append(i)  # 名詞が含まれたときにチャンクのインデックスをnounsに追加
    for i, j in combinations(nouns, 2):  # 名詞を含む文節のペア
        path_i = []
        path_j = []
        while i != j:
            if i < j:
                path_i.append(i)
                i = sentence.chunks[i].dst
            else:
                path_j.append(j)
                j = sentence.chunks[j].dst

        # 1つ目のケース．チャンクiから根までのパスにチャンクjがあるなら成り立つはず．
        if len(path_j) == 0:
            chunk_X = ''
            chunk_Y = ''
            for morph in sentence.chunks[path_i[0]].morphs:
                if morph.pos == '名詞':
                    chunk_X += 'X'
                elif morph.pos != '記号':
                    chunk_X += morph.surface

            for morph in sentence.chunks[i].morphs:
                if morph.pos == '名詞':
                    chunk_Y += 'Y'
                elif morph.pos != '記号':
                    chunk_Y += morph.surface

            chunk_X = re.sub('X+', 'X', chunk_X)  # 連結名詞もXで表現する
            chunk_Y = re.sub('Y+', 'Y', chunk_Y)  # Yも同様

            mid_path = []
            for n in path_i[1:]:
                for morph in sentence.chunks[n].morphs:
                    if morph.pos != '記号':
                        sf += morph.surface
                mid_path.append(sf)
                sf = ''

            path_XtoY = [chunk_X] + mid_path + [chunk_Y]
            print(' -> '.join(path_XtoY))

        # 2つ目のケース
        else:
            chunk_X = ''
            chunk_Y = ''
            for morph in sentence.chunks[path_i[0]].morphs:
                if morph.pos == '名詞':
                    chunk_X += 'X'
                elif morph.pos != '記号':
                    chunk_X += morph.surface

            for morph in sentence.chunks[path_j[0]].morphs:
                if morph.pos == '名詞':
                    chunk_Y += 'Y'
                elif morph.pos != '記号':
                    chunk_Y += morph.surface

            chunk_k = ''.join(
                [morph.surface if morph.pos != '記号' else '' for morph in sentence.chunks[i].morphs])  # 二つの名詞からdstを辿って重なったchunk
            chunk_X = re.sub('X+', 'X', chunk_X)
            chunk_Y = re.sub('Y+', 'Y', chunk_Y)

            mid_path_X = []
            mid_path_Y = []
            sf = ''
            for n in path_i[1:]:
                for morph in sentence.chunks[n].morphs:
                    if morph.pos != '記号':
                        sf += morph.surface
                mid_path_X.append(sf)
                sf = ''
            for n in path_j[1:]:
                for morph in sentence.chunks[n].morphs:
                    if morph.pos != '記号':
                        sf += morph.surface
                mid_path_Y.append(sf)
                sf = ''

            path_X = [chunk_X] + mid_path_X
            path_Y = [chunk_Y] + mid_path_Y

            print(' | '.join(
                [' -> '.join(path_X), ' -> '.join(path_Y), chunk_k]))
