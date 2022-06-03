from knock41 import sentences

flag1 = False  # 動詞が見つかったらTrue
flag2 = False  # 動詞に助詞がかかってたらTrue
verb = ''
cases = []

for sentence in sentences:
    for chunk in sentence.chunks:
        if len(chunk.srcs) > 0:
            for morph in chunk.morphs:
                if morph.pos == '動詞':
                    flag1 = True
                    verb = morph.base
                    break # 最左動詞をとるため
            for s in chunk.srcs:
                for morph in sentence.chunks[s].morphs:
                    if morph.pos == '助詞':
                        flag2 = True
                        cases.append(morph.surface)
        if flag1 and flag2:
            print(verb, ' '.join(cases), sep='\t')
        flag1 = False
        flag2 = False
        verb = ''
        cases.clear()
