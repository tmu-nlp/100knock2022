# knock49
# 文中のすべての名詞句のペアを結ぶ最短係り受けパスを抽出せよ．
# ただし，名詞句ペアの文節番号がiとj（i<j）のとき，係り受けパスは以下の仕様を満たすものとする．

# 1. 問題48と同様に，パスは開始文節から終了文節に至るまでの各文節の表現（表層形の形態素列）を” -> “で連結して表現する
# 2. 文節iとjに含まれる名詞句はそれぞれ，XとYに置換する

# また，係り受けパスの形状は，以下の2通りが考えられる．

# 1. 文節iから構文木の根に至る経路上に文節jが存在する場合: 文節iから文節jのパスを表示
# 2. 上記以外で，文節iと文節jから構文木の根に至る経路上で共通の文節kで交わる場合:
#    文節iから文節kに至る直前のパスと文節jから文節kに至る直前までのパス，文節kの内容を” | “で連結して表示

import knock41
from itertools import combinations
import re

for sentence in knock41.sentences:
    nouns = []
    #  名詞を含む文節を全て調査
    for i, chunk in enumerate(sentence.chunks):
        for morph1 in chunk.morphs:
            if morph1.pos == "名詞":
                nouns.append(i)
                break
    for i, j in combinations(nouns, 2):  # nounsから任意の2つの組み合わせを獲得
        # i,jは名詞句の節番号が格納
        path_i = []
        path_j = []
        # 小さい方からpath_{hoge}を計算して次の節に移動
        # i == j となった場合に終了
        while i != j:
            if i < j:
                path_i.append(i)
                i = sentence.chunks[i].dst
            else:
                path_j.append(j)
                j = sentence.chunks[j].dst
        if len(path_j) == 0:  # iから構文木への経路上にjがある場合
            # 文節iから文節jのパスを表示
            # 対応箇所を置換
            chunkX1 = ""  # 最初の名詞句path_i[0]
            chunkY1 = ""  # 最後の名詞句i
            # 名詞をXに置換する
            for morph2 in sentence.chunks[path_i[0]].morphs:
                if morph2.pos != "名詞" and morph2.pos != "記号":
                    chunkX1 += morph2.surface
                elif morph2.pos == "名詞":
                    chunkX1 += 'X'
            # 名詞をYに置換する
            for morph3 in sentence.chunks[i].morphs:
                if morph3.pos != "名詞" and morph3.pos != "記号":
                    chunkY1 += morph3.surface
                elif morph3.pos == "名詞":
                    chunkY1 += 'Y'
            chunkX1_sub = re.sub('X+', 'X', chunkX1)
            chunkY1_sub = re.sub('Y+', 'Y', chunkY1)
            # 経路上の文節を用意
            pathway = []
            for n in path_i[1:]:
                route_chunk = ""
                for morph4 in sentence.chunks[n].morphs:
                    if morph4.pos != "記号":
                        route_chunk += morph4.surface
                pathway.append(route_chunk)
            pathX2Y = [chunkX1_sub] + pathway + [chunkY1_sub]
            print(' -> '.join(pathX2Y))

        else:  # iとjが共通の文節kで交わる場合
            # 文節iから文節kに至る直前のパスと文節jから文節kに至る直前までのパス，文節kの内容を” | “で連結して表示
            chunkX2 = ""
            chunkY2 = ""
            chunk_k = ""  # 合流地点の文節
            for morph5 in sentence.chunks[path_i[0]].morphs:
                if morph5.pos != "名詞" and morph5.pos != "記号":
                    chunkX2 += morph5.surface
                elif morph5.pos == "名詞":
                    chunkX2 += 'X'
            for morph6 in sentence.chunks[path_j[0]].morphs:
                if morph6.pos != "名詞" and morph6.pos != "記号":
                    chunkY2 += morph6.surface
                elif morph6.pos == "名詞":
                    chunkY2 += 'Y'
            for morph7 in sentence.chunks[i].morphs:
                if morph7.pos != "記号":
                    chunk_k += morph7.surface
            chunkX2_sub = re.sub('X+', 'X', chunkX2)
            chunkY2_sub = re.sub('Y+', 'Y', chunkY2)
            pathwayX = []
            pathwayY = []
            path_X = []
            for n in path_i[1:]:
                route_chunk_x = ""
                for morph8 in sentence.chunks[n].morphs:
                    if morph8.pos != "記号":
                        route_chunk_x += morph8.surface
                pathwayX.append(route_chunk_x)
            for n in path_j[1:]:
                route_chunk_y = ""
                for morph9 in sentence.chunks[n].morphs:
                    if morph9.pos != "記号":
                        route_chunk_y += morph9.surface
                pathwayY.append(route_chunk_y)
            path_X = [chunkX2_sub] + pathwayX
            path_Y = [chunkY2_sub] + pathwayY
            print(' | '.join(
                [' -> '.join(path_X), ' -> '.join(path_Y), chunk_k]))
            # print(f"path_X : {path_X}")
            # print(f"path_Y : {path_Y}")
            # print(f"chunk_k : {chunk_k}")
