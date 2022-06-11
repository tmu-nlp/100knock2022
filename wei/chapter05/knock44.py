'''
44. 係り受け木の可視化
与えられた文の係り受け木を有向グラフとして、Graphviz等を使って可視化

'''
from knock41 import *
import pydot


if __name__ == '__main__':
    file_path = '../data/ai.ja.txt.parsed'
    sentences = get_chunks(file_path)[1]
    edges = []
    for chunk in sentences.chunks:

        if int(chunk.dst) == -1:
            continue
        else:
            # 係り元文節
            modifier = ''.join([morph.surface for morph in chunk.morphs if morph.pos != '記号'])
            # 係り先文節は係り先番号により取得
            modifiee = ''.join([morph.surface for morph in sentences.chunks[int(chunk.dst)].morphs if morph.pos != '記号'])
            edges.append((modifier, modifiee))


    n = pydot.Node('node')
    n.fontname = 'MS Gothic'
    n.fontsize = 9

    g = pydot.graph_from_edges(edges)
    g.add_node(n)
    g.write_jpeg('graph_from_edgds_dot.jpg', prog='dot')




# 出力のpngには文字化けだ。解決できないまま
# https://plaza.rakuten.co.jp/kugutsushi/diary/200711050001/
# https://srbrnote.work/archives/4205#toc9




