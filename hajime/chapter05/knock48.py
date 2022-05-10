# knock48
# 文中のすべての名詞を含む文節に対し，その文節から構文木の根に至るパスを抽出せよ．
# ただし，構文木上のパスは以下の仕様を満たすものとする．

# 1. 各文節は（表層形の）形態素列で表現する
# 2. パスの開始文節から終了文節に至るまで，各文節の表現を” -> “で連結する

# 「ジョン・マッカーシーはAIに関する最初の会議で人工知能という用語を作り出した。」という例文を考える
# CaboChaを係り受け解析に用いた場合，次のような出力が得られると思われる．

# ジョンマッカーシーは -> 作り出した
# AIに関する -> 最初の -> 会議で -> 作り出した
# 最初の -> 会議で -> 作り出した
# 会議で -> 作り出した
# 人工知能という -> 用語を -> 作り出した
# 用語を -> 作り出した

# アルゴリズム
# 名詞を含む文節を抽出する
# 対応する文節から

import knock41

for sentence in knock41.sentences:  # 一文を選択
    for chunk in sentence.chunks:
        for morph1 in chunk.morphs:
            if morph1.pos == "名詞":  # chunk.morphが名詞を含む文節である場合
                path = []
                base_noun = ""
                for morph2 in chunk.morphs:
                    if morph2.pos != "記号":
                        base_noun += morph2.surface
                path.append(base_noun)
                while chunk.dst != -1:
                    noun = ""
                    for morph3 in sentence.chunks[chunk.dst].morphs:
                        if morph3.pos != "記号":
                            noun += morph3.surface
                    path.append(noun)
                    chunk = sentence.chunks[chunk.dst]
                print(" -> ".join(path))
                break
