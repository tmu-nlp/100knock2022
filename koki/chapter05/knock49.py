'''
knock49 名詞間の係り受けパスの抽出

文中のすべての名詞句のペアを結ぶ最短係り受けパスを抽出せよ.
ただし, 名詞句ペアの文節番号がiとj（i<j）のとき, 係り受けパスは以下の仕様を満たすものとする. 

問題48と同様に, パスは開始文節から終了文節に至るまでの各文節の表現（表層形の形態素列）を” -> “で連結して表現する
文節iとjに含まれる名詞句はそれぞれ, XとYに置換する
また, 係り受けパスの形状は, 以下の2通りが考えられる.

パターン1 : 文節iから構文木の根に至る経路上に文節jが存在する場合: 文節iから文節jのパスを表示
パターン2 : 上記以外で, 文節iと文節jから構文木の根に至る経路上で共通の文節kで交わる場合: 文節iから文節kに至る直前のパスと文節jから文節kに至る直前までのパス, 文節kの内容を” | “で連結して表示

方針: 文中の名詞2つの組み合わせを総当たりで探索して、位置関係をパスで出力する
'''

from itertools import combinations  #リストの全組み合わせを取得
import re
import pandas as pd
from knock41 import sentences  # 係り受け解析結果の読み込み

sentence = sentences[1]  # 1行目の解析
lines = []  # 結果の全行を格納

'''名詞を含む文節のインデックスを抽出'''
nouns = []  # 名詞を含む文節の番号を格納
for idx, chunk in enumerate(sentence.chunks):
    for morph in chunk.morphs:
        if morph.pos == '名詞':
            nouns.append(idx)
            break  # 重複してappendしてしまうためbreakが必要

#print(len(nouns))  # [0, 1, 2, 3, 4, 6, 7, 9, 11, 12, 13, 15, 17, 18, 19, 20, 21, 22, 23, 25, 27, 28, 29, 30, 31, 32, 33, 34] 
#-> {}_28 C_2 = 378通り組み合わせのパスを出力

'''名詞を含む文節のペアごとにパスを作成'''
for i, j in combinations(nouns, 2):  # nounsから2つ選ぶ全組み合わせを取得して全探索
    path_i = []  # i起点のパスを格納
    path_j = []  # j起点のパスを格納
    
    path_I = ''
    path_J = ''
    path_ItoJ = ''
    chunk_I = ''
    chunk_J = ''

    while i != j:  #文節番号が i < j の時、共通のkが見つかった場合 = iがjに到達( i == j )するまでサーチ
        if i < j:
            path_i.append(i)   
            i = sentence.chunks[i].dst  # 係り先へ
        else:  # i起点のペアが無くなった場合にj起点のペア探索へ移る (j < i)
            path_j.append(j)
            j = sentence.chunks[j].dst

    line = ''  # 出力用の結果1行を格納、初期化

    #パターン1: 文節iから構文木の根に至る経路上に文節jが存在する場合
    if len(path_j) == 0:  # j始点でない場合, 上記whuile文でi始点で途中にjが含まれる場合ループが終了する
        
        #文節iをXに変えてchunk_Iに格納
        for morph in sentence.chunks[path_i[0]].morphs:  # path_i[0]はpathの始点
            if morph.pos == '名詞':
                chunk_I += 'X'
            elif morph.pos != '記号':
                chunk_I += morph.surface
                #chunk_X.append(morph.surface)

        #文節jをYに変えてchunk_Jに格納
        for morph in sentence.chunks[i].morphs:
            if morph.pos == '名詞':
                chunk_J += 'Y'  # 名詞句をYとして格納
            elif morph.pos != '記号':
                chunk_J += morph.surface

        chunk_I = re.sub('X+', 'X', chunk_I)  # 人工 + 知能 とかの名詞が連接する場合(XX -> X)とする?
        chunk_J = re.sub('Y+', 'Y', chunk_J)

        #文節iから文節jに至る途中のパス
        mid_path = [] # 文節毎に -> が必要なためリストで最後にjoin
        for n in path_i[1:]:  #始点は含まない
             for morph in sentence.chunks[n].morphs:
                 if morph.pos != '記号':
                    mid_path.append(morph.surface)

        path_ItoJ += chunk_I + ' -> ' + ' -> '.join(mid_path) + ' -> ' + chunk_J
        line += ' -> ' + path_ItoJ

    #パターン2: 文節iから構文木の根に至る経路上に文節jが存在しない場合
    else:
        chunk_k = ''
        # 文節i
        for morph in sentence.chunks[path_i[0]].morphs:
            if morph.pos == '名詞':
                chunk_I += 'X'
            elif morph.pos != '記号':
                chunk_I += morph.surface
            
        chunk_I = re.sub('X+', 'X', chunk_I)

        #文節j
        for morph in sentence.chunks[path_j[0]].morphs:
            if morph.pos == '名詞':
                chunk_J += 'Y'
            elif morph.pos != '記号':
                chunk_J += morph.surface
            
        chunk_J = re.sub('Y+', 'Y', chunk_J)

        #文節k
        # 上記 while i == j で終了している, すなわち共通の文節kのインデックス番号に他ならない
        for morph in sentence.chunks[i].morphs:
            if morph.pos != '記号':
                chunk_k += morph.surface

        #文節iの次の文節から文節kに至る途中のパス (path_Iで言うところのインデックス1から)
        mid_path_i = []
        mid_path_i_line = ''
        tmp_mid = []
        if(len(path_i) > 3):
            for n in path_i[1:]:
                for morph in sentence.chunks[n].morphs:
                    if morph.pos != '記号':
                        tmp_mid.append(morph.surface)  # 表層形を格納
                mid_path_i.append(''.join(tmp_mid))  # 表層形 -> 文節
                tmp_mid = []  # 初期化
            mid_path_i_line = ' -> '.join(mid_path_i)

        #文節jの次の文節から文節kに至る途中のパス (path_Jで言うところのインデックス1から)
        mid_path_j = []
        mid_path_j_line = ''  # 出力形式
        tmp_mid = []
        if(len(path_j) > 3):
            for n in path_j[1:]:
                    for morph in sentence.chunks[n].morphs:
                        if morph.pos != '記号':
                            tmp_mid.append(morph.surface)
                    mid_path_j.append(''.join(tmp_mid))
                    tmp_mid = []
            mid_path_j_line = ' -> '.join(mid_path_j)

        #文節iから文節kに至る途中のパス
        path_I += chunk_I + ' -> ' + mid_path_i_line
        #文節jから文節kに至る途中のパス
        path_J += chunk_J + ' -> ' + mid_path_j_line
        
        line += path_I + ' | ' + path_J + ' | ' + chunk_k

    lines.append(line)

series = pd.Series(lines, index = None)
series.to_csv('./result/output49.txt', index=None, header=None)