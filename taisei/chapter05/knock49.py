from knock41 import Chunk
from knock41 import Morph
from knock41 import chunk_list_list
from knock42 import get_chunk_phrase
from knock43 import get_chunk_phrase_pos
import networkx as nx #グラフ構造を作成したい(pathの探索くらいだから別に使わなくてもよかったって途中で気づいた)
from itertools import combinations

def make_graph(chunk_list2): #文ごとにグラフ構造を作成 使用したライブラリ以外はknock44とほとんど同じ
    dg = nx.DiGraph()
    for i, chunk_obj in enumerate(chunk_list2):
        text_from = get_chunk_phrase(chunk_obj)
        ans_text1 = f'{text_from} {i}' #グラフ上で同じ言葉をひとつにまとめちゃうから、インデックス番号をつけることで回避
        dg.add_node(ans_text1)

        if chunk_obj.dst != -1:
            text_for = get_chunk_phrase(chunk_list2[chunk_obj.dst])
            ans_text2 = f'{text_for} {chunk_obj.dst}'
            dg.add_node(ans_text2)
            dg.add_edge(ans_text1, ans_text2)
    return dg


def get_root(chunk_list2, chunk_obj3): #引数chunk_obj3の根のchunkオブジェクトとその文節番号を取得
    num = -1
    while (chunk_obj3.dst != -1): #係先がなくなるまで
        num = chunk_obj3.dst   
        chunk_obj3 = chunk_list2[chunk_obj3.dst]  
    return chunk_obj3, num


def get_noun_list(chunk_list2): #引数chunk_list2で名詞を含む文節とその文節番号を取得
    noun_list = [] #(名詞を含む文節, その文節番号)　っていうタプルを要素に持つリスト
    for i, chunk_obj2 in enumerate(chunk_list2):
        new_str = get_chunk_phrase_pos(chunk_obj2, "名詞")
        if new_str != "":
            noun_list.append((new_str, i))
    return noun_list


def replace_noun(chunk_list2, num, word): #文節内で名詞句（1つor複数連続した名詞）をwordに置換。名詞句が2つ以上の場合はそれぞれ置換
    cnt = False #名詞句を探すために名詞が連続しているかの判定
    chunk_obj2 = chunk_list2[num]
    for_phrase = []
    for morph in chunk_obj2.morphs:
        if morph.pos == "名詞":
            if cnt == False:
                for_phrase.append(word)
                cnt = True
            else:
                continue
        elif morph.pos != "記号":
            for_phrase.append(morph.surface)
            cnt = False
    return "".join(for_phrase)


with open("./output/knock49_output.txt", "w") as f_out:
    for chunk_list in chunk_list_list:
        noun_list = get_noun_list(chunk_list)
        dg = make_graph(chunk_list)
        combi_noun_list = list(combinations(noun_list, 2)) #noun_listの要素のペアのリスト

        for combi in combi_noun_list: #名詞のペアごとに処理
            #print(combi[0], combi[1]) 名詞句X, 名詞句Y
            chunk_root_x , num_x = get_root(chunk_list, chunk_list[combi[0][1]])
            chunk_root_y , num_y = get_root(chunk_list, chunk_list[combi[1][1]])
            if num_y == -1: #着目してる2つ目の名詞が根の時(source == targetの時はなぜかpathが空になってしまう仕様みたい)
                path_x = list(nx.all_simple_paths(dg, source=f'{combi[0][0]} {combi[0][1]}', target=f'{get_chunk_phrase(chunk_root_x)} {num_x}'))[0]
                path_y = [f'{combi[1][0]} {combi[1][1]}']
            else:
                path_x = list(nx.all_simple_paths(dg, source=f'{combi[0][0]} {combi[0][1]}', target=f'{get_chunk_phrase(chunk_root_x)} {num_x}'))[0] #sourceからtargetまでへのpath(sourceとtargetも含む)
                path_y = list(nx.all_simple_paths(dg, source=f'{combi[1][0]} {combi[1][1]}', target=f'{get_chunk_phrase(chunk_root_y)} {num_y}'))[0] 
                #nx.all_simple_pathsのリスト化は[[分節 分節番号, 分節 分節番号, 分節 分節番号, ...]]と言う二重のリストになってるから[0]としてpath_xらに格納
            same_start = ""
            text_list_x = []
            text_list_y = []
            text_same = ""
            for noun_num in path_x: #path_xとpath_yの共通部分の開始の文節を探す
                if noun_num in path_y:
                    same_start = noun_num
                    break
            
            if same_start == f'{combi[1][0]} {combi[1][1]}': #path_yがpath_xの1部の場合
                for noun_now in path_x:
                    if noun_now != same_start:
                        if noun_now == path_x[0]:
                            text_list_x.append(replace_noun(chunk_list, int(noun_now.split()[1]), "X"))
                        else:
                            text_list_x.append(noun_now.split()[0])
                    else:
                        text_list_x.append(replace_noun(chunk_list, int(noun_now.split()[1]), "Y"))
                        break
                f_out.write(f'{" -> ".join(text_list_x)}\n')


            else: #path_xとpath_yが途中で合流する場合
                for noun_now in path_x:
                    if noun_now != same_start:
                        if noun_now == path_x[0]:
                            text_list_x.append(replace_noun(chunk_list, int(noun_now.split()[1]), "X"))
                        else:
                            text_list_x.append(noun_now.split()[0])
                    else:
                        text_same = noun_now.split()[0]
                        break

                for noun_now in path_y:
                    if noun_now != same_start:
                        if noun_now == path_y[0]:
                            text_list_y.append(replace_noun(chunk_list, int(noun_now.split()[1]), "Y"))
                        else:
                            text_list_y.append(noun_now.split()[0])
                    else:
                        break

                str_x = " -> ".join(text_list_x)
                str_y = " -> ".join(text_list_y)
                f_out.write(f'{" | ".join([str_x, str_y, text_same])}\n')
