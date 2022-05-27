from knock41 import sentences
from graphviz import Digraph

g = Digraph(filename='44')
for sentence in sentences:
    for id, chunk in enumerate(sentence.chunks):
        if chunk.dst != -1:
            modifier = ''.join([morph.surface if morph.pos !=
                               '記号' else '' for morph in chunk.morphs] + ['(' + str(id) + ')'])
            modifiee = ''.join([morph.surface if morph.pos !=
                               '記号' else '' for morph in sentence.chunks[chunk.dst].morphs] + ['(' + str(chunk.dst) + ')'])
            g.node(modifier)
            g.node(modifiee)
            g.edge(modifier, modifiee)
g.render(view=True)
