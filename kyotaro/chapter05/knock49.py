"""
文中のすべての名詞句のペアを結ぶ最短係り受けパスを抽出せよ. ただし, 名詞句ペアの文節番号がiとj（i<j）のとき, 係り受けパスは以下の仕様を満たすものとする
"""
import sys
from common import Morph
from common import Chunk
from common import Sentence
from common import set_matrioshk

file_name = sys.argv[1]
sentences, chunks, morphs = set_matrioshk(file_name)


for sentence in sentences:
    for i in range(len(sentence.chunks)):
        for j in range(i + 1, len(sentence.chunks)):
            x_chunk = sentence.chunks[i]
            x = list()
            x_pos = list()
            x_modifee = list()
            x_ans = list()
            x_add = list()

            y_chunk = sentence.chunks[j]
            y = list()
            y_pos = list()
            y_modifee = list()
            y_ans = list()
            y_add = list()

            for x_morph in x_chunk.morphs:
                x_pos.append(x_morph.pos)
            if "名詞" in x_pos:
                for x_morph in x_chunk.morphs:
                    if x_morph.pos != "記号":
                        if x_morph.pos != "名詞":
                            x_add.append(x_morph.surface)
                        else:
                            x.append(x_morph.surface)
                        while x_chunk.dst != -1:
                            x_modifee.append("".join(morph.surface for morph in sentence.chunks[x_chunk.dst].morphs if morph.pos != "記号"))
                            x_chunk = sentence.chunks[x_chunk.dst]
                if x and x_modifee:
                    x = ["".join(x)]
                    x_ans = x + x_modifee

            for y_morph in y_chunk.morphs:
                y_pos.append(y_morph.pos)
            if "名詞" in y_pos:
                for y_morph in y_chunk.morphs:
                    if y_morph.pos != "記号":
                        if y_morph.pos != "名詞":
                            y_add.append(y_morph.surface)
                        else:
                            y.append(y_morph.surface)
                        while y_chunk.dst != -1:
                            y_modifee.append("".join(morph.surface for morph in sentence.chunks[y_chunk.dst].morphs if morph.pos != "記号"))
                            y_chunk = sentence.chunks[y_chunk.dst]
                if x and x_modifee and y and y_modifee:
                    y = ["".join(y)]
                    y_ans = y + y_modifee
            
            # print(x_ans)
            # print(y_ans)
            x_add = "".join(x_add)
            y_add = "".join(y_add)
            if x_ans and y_ans:
                n = ""
                m = ""
                for a in range(len(x_ans)):
                    for b in range(len(y_ans)):
                        if x_ans[a] == y_ans[b]:
                            n = x_ans[:a]
                            m = y_ans[:b]
                            n_left = x_ans[a-1:]
                            break
                    if n and m:
                        if len(n) == 1 and len(m) == 1:
                            print("X" + x_add + " | Y" + y_add + " | " + x_ans[a])
                        elif len(n) > 1 and len(m) == 1:
                            if n_left == y_ans:
                                n = " -> ".join(n[1:len(n) - 1])
                                if not(n):
                                    print("X" + x_add + " -> Y" + y_add)
                                else:
                                    print("X" + x_add + " -> " + n + " -> Y" + y_add)
                            else:
                                n = " -> ".join(n[1:])
                                print("X" + x_add + " -> " + n + " | " + "Y" + y_add + " | " + x_ans[a])
                        elif len(n) == 1 and len(m) > 1:
                            if n_left == y_ans:
                                print("X" + x_add + " -> Y" + y_add)
                            else:
                                m = " -> ".join(m[1:])
                                print("X" + x_add + " | " + "Y" + y_add + " -> " + m + " | " + x_ans[a])
                        else:
                            n = " -> ".join(n[1:])
                            m = " -> ".join(m[1:])
                            print("X" + x_add + " -> " + n + " | " + "Y" + y_add + " -> " + m + " | " + x_ans[a])
                        break
