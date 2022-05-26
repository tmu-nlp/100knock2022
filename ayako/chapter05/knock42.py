#knock42 係り元と係り先の文節の表示
#係り元の文節と係り先の文節のテキストをタブ区切り形式ですべて抽出せよ．
#ただし，句読点などの記号は出力しないようにせよ．
import knock41

fname = "ai.ja.txt.parsed"
text = knock41.parse_chunk(fname)
for sentence in text[1:]:#タイトルは除く
    chunks = []
    dsts = []
    for chunk in sentence:
        string = ""
        for morph in chunk.morphs:
            if morph.pos != "記号":
                string += morph.surface
        chunks.append(string)#文節テキスト保存
        dsts.append(chunk.dst)#係先の文節番号保存

    for i in range(len(chunks)):
        if dsts[i] != -1:
            print(chunks[i]+"\t"+chunks[int(dsts[i])])
            