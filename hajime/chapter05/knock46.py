# knock46
# 45のプログラムを改変し，述語と格パターンに続けて項（述語に係っている文節そのもの）をタブ区切り形式で出力せよ．
# 45の仕様に加えて，以下の仕様を満たすようにせよ．

# 1. 項は述語に係っている文節の単語列とする（末尾の助詞を取り除く必要はない）
# 2. 述語に係る文節が複数あるときは，助詞と同一の基準・順序でスペース区切りで並べる

# 「ジョン・マッカーシーはAIに関する最初の会議で人工知能という用語を作り出した。」という例文を考える．
# この文は「作り出す」という１つの動詞を含み，
# 「作り出す」に係る文節は「ジョン・マッカーシーは」，「会議で」，「用語を」であると解析された場合は，
# 次のような出力になるはずである．

# 作り出す	で は を	会議で ジョンマッカーシーは 用語を


import knock41

for sentence in knock41.sentences:  # 一文を選択
    for chunk in sentence.chunks:
        for morph1 in chunk.morphs:
            if morph1.pos == '動詞':
                part = []
                frame = ""
                for src in chunk.srcs:
                    for morph2 in sentence.chunks[src].morphs:
                        if morph2.pos == '助詞':
                            frame = ""
                            for morph3 in sentence.chunks[src].morphs:
                                if morph3.pos != '記号':
                                    frame += morph3.surface
                            part.append((morph2.surface, frame))
                if len(part) > 0:
                    sort_part = sorted(list(set(part)), key=lambda x: x[0])
                    pattern_case = ""
                    frame_case = ""
                    for row in sort_part:
                        pattern_case += row[0] + " "
                        frame_case += row[1] + " "
                    frame_case = frame_case.rstrip(" ")
                    line = morph1.base + "\t" + pattern_case + frame_case
                    print(line)
                break  # このbreakで最左のみを処理している
