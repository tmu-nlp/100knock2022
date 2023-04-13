# knock45 動詞の格パターンの抽出
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
            parts = []  # 助詞のリスト
            for src in chunk.srcs:
                for morph in sentence[src].morphs:  # 文節に助詞を含むかチェック
                    if morph.pos == "助詞":
                        parts.append(morph.surface)
            if len(parts) > 0:
                parts = sorted(parts)  # 辞書順に出力
                parts = " ".join(parts)  # 助詞を空白区切りで出力
                print(f"{verb}\t{parts}")