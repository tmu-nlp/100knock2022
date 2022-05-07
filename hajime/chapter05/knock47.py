# knock47
# 動詞のヲ格にサ変接続名詞が入っている場合のみに着目したい．
# 46のプログラムを以下の仕様を満たすように改変せよ．

# 1. 「サ変接続名詞+を（助詞）」で構成される文節が動詞に係る場合のみを対象とする
# 2. 述語は「サ変接続名詞+を+動詞の基本形」とし，文節中に複数の動詞があるときは，最左の動詞を用いる
# 3. 述語に係る助詞（文節）が複数あるときは，すべての助詞をスペース区切りで辞書順に並べる
# 4. 述語に係る文節が複数ある場合は，すべての項をスペース区切りで並べる（助詞の並び順と揃えよ）

# 例えば「また、自らの経験を元に学習を行う強化学習という手法もある。」という文から，以下の出力が得られるはずである．
# 学習を行う	に を	元に 経験を

# アルゴリズム
# 最左動詞を発見
# 最左動詞に「サ変接続名詞+を」で構成されている文節がかかっているか判断
# かかっていたら，述語として「サ変接続名詞+を+最左動詞の基本形」を抽出
# 指定の述語にかかる助詞/文節を辞書順に並べる


import knock41

for sentence in knock41.sentences:  # 一文を選択
    for chunk in sentence.chunks:
        for morph1 in chunk.morphs:
            if morph1.pos == '動詞':  # 最左動詞を発見
                for i, src1 in enumerate(chunk.srcs):  # 動詞にかかっている文節を調査
                    if len(sentence.chunks[src1].morphs) == 2 and sentence.chunks[src1].morphs[0].pos1 == "サ変接続" and sentence.chunks[src1].morphs[1].surface == "を":
                        # 長さが2で「サ変接続名詞+を」の形になっているかどうか判断
                        trg_pred = sentence.chunks[src1].morphs[0].surface + \
                            sentence.chunks[src1].morphs[1].surface + \
                            morph1.base
                        # 述語の準備が完了
                        part = []
                        frame = ""
                        num = 0
                        for src2 in chunk.srcs[:i] + chunk.srcs[i+1:]:  # i番目は抜かす
                            for morph2 in sentence.chunks[src2].morphs:
                                if morph2.pos == "助詞":
                                    frame = ""
                                    for morph3 in sentence.chunks[src2].morphs:
                                        if morph3.pos != '記号':
                                            frame += morph3.surface
                                    part.append((morph2.surface, frame))
                        if len(part) > 0:
                            sort_part = sorted(
                                list(set(part)), key=lambda x: x[0])
                            pattern_case = ""
                            frame_case = ""
                            for row in sort_part:
                                pattern_case += row[0] + " "
                                frame_case += row[1] + " "
                            frame_case = frame_case.rstrip(" ")
                            line = trg_pred + "\t" + pattern_case + "\t" + frame_case
                            print(line)
                break
