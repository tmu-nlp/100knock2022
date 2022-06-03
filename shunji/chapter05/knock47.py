from unittest import case
from knock41 import sentences

flag1 = False  # 動詞が見つかったらTrue
flag2 = False  # 動詞にサ変+をでTrue
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
                src_morphs = sentence.chunks[s].morphs
                for i, morph in enumerate(src_morphs):
                    if morph.pos == '助詞':
                        cases.append(morph.surface)
                        # argsに項を追加
                        args.append(
                            ''.join([morph.surface for morph in sentence.chunks[s].morphs])
                        )
                    if i < len(src_morphs)-1 and morph.pos1 == 'サ変接続' and src_morphs[i+1].surface == 'を':
                        flag2 = True
            if flag1 and flag2:
                print(verb, ' '.join(cases), ' '.join(args), sep='\t')  # 項も出力
            flag1 = False
            flag2 = False
            verb = ''
            cases.clear()
            args.clear()  # リセット
