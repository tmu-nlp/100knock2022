from knock41 import sentences

for sentence in sentences:
    for chunk in sentence.chunks:
        if chunk.dst != -1:
            modifier = ''.join(
                [morph.surface if morph.pos != '記号' else '' for morph in chunk.morphs])
            modifiee = ''.join([morph.surface if morph.pos !=
                               '記号' else '' for morph in sentence.chunks[chunk.dst].morphs])
            print(modifier, modifiee, sep='\t')
