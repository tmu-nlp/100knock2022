## knock44.py 係り受け木の可視化

#与えられた文の係り受け木を有効グラフとして可視化せよ。可視化にはGraphviz等を用いるとよい。

from knock41 import sentences
from graphviz import Digraph

for idx, sentence in enumerate(sentences):
    g = Digraph(format='png')

    for chunk in sentence.chunks:
        if chunk.dst != -1:
            modifiers = [] #係り先の表層系を格納
            modifiees = [] #係り元の表層系を格納

            #掛かり先
            for morph in chunk.morphs:
                if morph.pos != '記号':
                    modifier = morph.surface
                    modifiers.append(modifier)

            #係り元
            for morph in sentence.chunks[chunk.dst].morphs: #掛かり先番号を指定して、係り元の情報を取り出す
                if morph.pos != '記号':
                    modifiee = morph.surface
                    modifiees.append(modifiee) 

            modifer = ''.join(modifiers)     
            modifee = ''.join(modifiees)
            g.edge(modifer, modifee)
    
    if idx == 1: #ファイルが多いため1行目のみ可視化
        g.render(directory='./result', filename='output44-{}'.format(str(idx)))