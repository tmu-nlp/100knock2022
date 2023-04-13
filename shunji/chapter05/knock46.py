from knock41 import sentences

flag1 = False  # 動詞が見つかったらTrue
flag2 = False  # 動詞に助詞がかかってたらTrue
verb = ''
cases = []
args = []

for sentence in sentences:
    for chunk in sentence.chunks:
        if len(chunk.srcs) > 0:
            for morph in chunk.morphs:
                if morph.pos == '動詞':
                    flag1 = True
                    verb = morph.base
            for s in chunk.srcs:
                for morph in sentence.chunks[s].morphs:
                    if morph.pos == '助詞':
                        flag2 = True
                        cases.append(morph.surface)
                        # argsに項を追加
                        args.append(
                            ''.join([morph.surface if morph.pos != '記号' else '' for morph in sentence.chunks[s].morphs])
                        )
        if flag1 and flag2:
            print(verb + '\t' + ' '.join(cases + args))  # 項も出力
        flag1 = False
        flag2 = False
        verb = ''
        cases.clear()
        args.clear()  # リセット
