# knock48 名詞から根へのパスの抽出

# 文中のすべての名詞を含む文節に対し，その文節から構文木の根に至るパスを抽出せよ． 
# ただし，構文木上のパスは以下の仕様を満たすものとする．

from knock41 import sentences

with open('./result/output48.txt', 'w') as f:
    sentence = sentences[1]  #1行目の解析
    base_path = ''  # 根の親となる名詞を一時格納
    path = []  # パス(全体)をカンマ区切りで格納

    for chunk in sentence.chunks:
        #構文木の親となる名詞を発見、格納
        if '名詞' in [morph.pos for morph in chunk.morphs]:  # 文節が名詞を含むか
            for morph in chunk.morphs:
                if morph.pos != '記号':
                    base_path += morph.surface  # 文節中の(キーワードの)名詞を加えていく  
            path.append(base_path)  # 名詞(キーワード)を格納
            base_path = ''

            #構文木の親から根まで解析
            while chunk.dst != -1:
                tmp_path = ''  # 根の子情報を一時格納
                for morph in sentence.chunks[chunk.dst].morphs:
                    if morph.pos != '記号':
                        tmp_path += morph.surface
                chunk = sentence.chunks[chunk.dst]  # chunkを係り先へ更新
                path.append(tmp_path)
            print(' -> '.join(path), file = f)
            path = []  # 初期化

