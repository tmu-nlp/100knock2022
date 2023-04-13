'''"
49. 名詞間の係り受けパスを抽出
文中の全ての名詞句のペアを結ぶ最短係り受けパスを抽出し、
名詞句ペアの文節番号がiとj(i<j)の場合、係り受けパスは以下の仕様を満たす
・パスを表現するには、開始文節から終了文節に至るまでの各文節(表層形の形態素列)を'->'で連結
・文節iとjに含まれる名詞句はそれぞれ、XとYに置換
・文節iとjから根に至る途中で文節ｋで交わる場合、iからｋまでのパス | jからkまでのパス | kの内容
'''""

from knock41 import *
import re
from itertools import combinations


if __name__ == '__main__':
    file_path = '../data/ai.ja.txt.parsed'
    sentences = get_chunks(file_path)
    sentence = sentences[1]
    nouns = []
    for i,chunk in enumerate(sentence.chunks):
        if [morph for morph in chunk.morphs if morph.pos == '名詞']:
            nouns.append(i)
    for i, j in combinations(nouns, 2):
        path_i = []
        path_j =[]
        while i != j:
            if i < j:
                path_i.append(i)
                i = sentence.chunks[i].dst
            else:
                path_j.append(j)
                j = sentence.chunks[j].dst


        if len(path_j) == 0:
            X = 'X' + ''.join([morph.surface for morph in sentence.chunks[path_i[0]].morphs if morph.pos != '名詞' and morph.pos != '記号'])
            Y = 'Y' + ''.join([morph.surface for morph in sentence.chunks[i].morphs if morph.pos != '名詞' and morph.pos != '記号'])
            chunk_X = re.sub('X+', 'X', X)
            chunk_Y = re.sub('Y+', 'Y', Y)
            path_iTOj = [chunk_X] + [''.join(morph.surface for n in path_i[1:] for morph in sentence.chunks[n].morphs)] + [chunk_Y]
            print('->'.join(path_iTOj))
        else:
            X = 'X' + ''.join([morph.surface for morph in sentence.chunks[path_i[0]].morphs if morph.pos != '名詞' and morph.pos != '記号'])
            Y = 'Y' + ''.join([morph.surface for morph in sentence.chunks[path_j[0]].morphs if morph.pos != '名詞' and morph.pos != '記号'])

            chunk_X = re.sub('X+', 'X', X)
            chunk_Y = re.sub('Y+', 'Y', Y)
            chunk_k = ''.join([morph.surface for morph in sentence.chunks[i].morphs if morph.pos != '記号'])
            path_X = [chunk_X] + [''.join(morph.surface for n in path_i[1:] for morph in sentence.chunks[n].morphs if morph.pos != '記号')]
            path_Y = [chunk_Y] + [''.join(morph.surface for n in path_j[1:] for morph in sentence.chunks[n].morphs if morph.pos != '記号')]
            print(' | '.join(['->'.join(path_X), '->'.join(path_Y), chunk_k]))


