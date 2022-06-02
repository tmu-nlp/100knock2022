from knock41 import sentences

flag1 = False
flag2 = False
modifier = []
modifiee = []

for sentence in sentences:
    for chunk in sentence.chunks:
        if chunk.dst != -1:
            for morph in chunk.morphs:
                if morph.pos == '名詞':
                    flag1 = True
                if morph.pos != '記号':
                    modifier.append(morph.surface)
            for morph in sentence.chunks[chunk.dst].morphs:
                if morph.pos == '動詞':
                    flag2 = True
                if morph.pos != '記号':
                    modifiee.append(morph.surface)
            if flag1 and flag2:
                print(''.join(modifier), ''.join(modifiee), sep='\t')
            flag1 = False
            flag2 = False
            modifier.clear()
            modifiee.clear()