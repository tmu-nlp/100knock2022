# knock46
# 45のプログラムを改変し，述語と格パターンに続けて項（述語に係っている文節そのもの）をタブ区切り形式で出力せよ．
# 45の仕様に加えて，以下の仕様を満たすようにせよ．
# 項は述語に係っている文節の単語列とする（末尾の助詞を取り除く必要はない）
# 述語に係る文節が複数あるときは，助詞と同一の基準・順序でスペース区切りで並べる
import knock41

fname = "ai.ja.txt.parsed"
text = knock41.parse_chunk(fname)

for sentence in text:
    for chunk in sentence:
        verb = ""
        for morph in chunk.morphs:  # 文節に動詞を含むかチェック
            if morph.pos == "動詞":
                verb += morph.base  # 最左の動詞（一番最初の動詞）の基本形を述語にする
                break
        if len(verb) > 0:  # 動詞を含む文節の時
            particle_list = []  # [助詞，文節]のリスト
            for src in chunk.srcs:
                surface = ""  # 述語にかかる文節
                for morph in sentence[src].morphs:  # 文節に助詞を含むかチェック
                    if morph.pos == "助詞":
                        surface = "".join(
                            [m.surface for m in sentence[src].morphs if m.pos != "記号"])  # 記号除く
                        particle_list.append([morph.base, surface])
            parts = []  # 助詞のリスト
            chunks = []  # 文節のリスト
            if len(particle_list) > 0:  # 係元に助詞ない奴は出力しない
                for particle in sorted(particle_list):  # 助詞の辞書順に出力
                    parts.append(particle[0])
                    chunks.append(particle[1])

                parts = " ".join(parts)  # 助詞を空白区切りで出力
                chunks = " ".join(chunks)  # 文節を空白区切りで出力
                print(f"{verb}\t{parts}\t{chunks}")
