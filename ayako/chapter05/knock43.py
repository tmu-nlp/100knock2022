# knock43 名詞を含む文節が動詞を含む文節に係るものを抽出
# 名詞を含む文節が，動詞を含む文節に係るとき，これらをタブ区切り形式で抽出せよ．
# ただし，句読点などの記号は出力しないようにせよ．
import knock41

fname = "ai.ja.txt.parsed"
text = knock41.parse_chunk(fname)
for sentence in text[1:]:
    chunks = []
    dsts = []
    nouns = []
    verbs = []
    for chunk in sentence:
        string = ""
        noun = False
        verb = False
        for morph in chunk.morphs:
            if morph.pos != "記号":
                string += morph.surface
            if morph.pos == "名詞":
                noun = True
            if morph.pos == "動詞":
                verb = True
        chunks.append(string)  # 文節テキスト保存
        dsts.append(chunk.dst)  # 係先の文節番号保存
        nouns.append(noun)
        verbs.append(verb)
    for i in range(len(chunks)):
        if dsts[i] != -1:
            if nouns[i] and verbs[dsts[i]]:
                print(chunks[i] + "\t" + chunks[int(dsts[i])])
