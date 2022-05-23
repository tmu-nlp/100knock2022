from knock41 import Chunk
from knock41 import Morph
from knock41 import chunk_list_list
from knock42 import get_chunk_phrase
from graphviz import Digraph

for j, chunk_list in enumerate(chunk_list_list):
    dg = Digraph(format='png')
    for i, chunk_obj in enumerate(chunk_list):
        text_from = get_chunk_phrase(chunk_obj)
        ans_text1 = f'{text_from} {i}' #グラフ上で同じ言葉をひとつにまとめちゃうから、インデックス番号をつけることで回避
        dg.node(ans_text1)

        if chunk_obj.dst != -1:
            text_for = get_chunk_phrase(chunk_list[chunk_obj.dst])
            ans_text2 = f'{text_for} {chunk_obj.dst}'
            dg.node(ans_text2)
            dg.edge(ans_text1, ans_text2)

    dg.render('./output/knock44_output/dgraph' + str(j))