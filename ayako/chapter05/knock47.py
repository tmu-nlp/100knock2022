# knock47
# 動詞のヲ格にサ変接続名詞が入っている場合のみに着目したい．
# 46のプログラムを以下の仕様を満たすように改変せよ．
# 「サ変接続名詞+を（助詞）」で構成される文節が動詞に係る場合のみを対象とする
# 述語は「サ変接続名詞+を+動詞の基本形」とし，文節中に複数の動詞があるときは，最左の動詞を用いる
# 述語に係る助詞（文節）が複数あるときは，すべての助詞をスペース区切りで辞書順に並べる
# 述語に係る文節が複数ある場合は，すべての項をスペース区切りで並べる（助詞の並び順と揃えよ）
import knock41

fname = "ai.ja.txt.parsed"
text = knock41.parse_chunk(fname)

for sentence in text:
    for i, chunk in enumerate(sentence):
        verb = ""
        for morph in chunk.morphs:  # 文節に動詞を含むかチェック
            if morph.pos == "動詞":
                verb += morph.base  # 最左の動詞（一番最初の動詞）の基本形を述語にする
                break
        if len(verb) > 0:  # 動詞含む文節の時
            particle_list = []  # [助詞，文節]のリスト
            for src in chunk.srcs:
                if len(sentence[src].morphs) == 2 and sentence[src].morphs[0].pos1 == "サ変接続" and sentence[src].morphs[1].base == "を":
                    verbs = sentence[src].morphs[0].surface + \
                        sentence[src].morphs[1].surface + verb
                    surface = ""  # 述語にかかる文節のうち「サ変＋を」以外
                    for src2 in chunk.srcs:
                        if src2 == src:  # 「サ変＋を」以外
                            continue
                        for morph in sentence[src2].morphs:  # 文節に助詞を含むかチェック
                            if morph.pos == "助詞":
                                surface = "".join(
                                    [m.surface for m in sentence[src2].morphs if m.pos != "記号"])  # 記号除く
                                particle_list.append([morph.base, surface])
                    parts = []  # 助詞のリスト
                    chunks = []  # 文節のリスト
                    if len(particle_list) > 0:  # 係元に助詞ない奴は出力しない
                        for particle in sorted(particle_list):  # 助詞の辞書順に出力
                            parts.append(particle[0])
                            chunks.append(particle[1])

                        parts = " ".join(parts)  # 助詞を空白区切りで出力
                        chunks = " ".join(chunks)  # 文節を空白区切りで出力
                        print(f"{verbs}\t{parts}\t{chunks}")
