#knock48
#文中のすべての名詞を含む文節に対し，その文節から構文木の根に至るパスを抽出せよ． 
# ただし，構文木上のパスは以下の仕様を満たすものとする．
#各文節は（表層形の）形態素列で表現する
#パスの開始文節から終了文節に至るまで，各文節の表現を” -> “で連結する
import knock41

def search_path(sentence, chunk, result):
    """ある文節スタートで根までパスをたどり，通ったノードをresultに格納する"""
    if chunk.dst == -1:#係先がなくなった時終わり
        return result
    result.append(sentence[chunk.dst])#係先の文節を追加
    return search_path(sentence, sentence[chunk.dst], result)#係先の文節にも同じ操作

def main(text):
    for sentence in text:
        for chunk in sentence:
            noun_flag = False
            for morph in chunk.morphs:  # 文節に名詞を含むかチェック
                if morph.pos == "名詞":
                    noun_flag = True
            if not noun_flag:
                continue
            result = search_path(sentence, chunk, [chunk])
            ans = []
            for chunk in result:
                surface = "".join([m.surface for m in chunk.morphs if m.pos != "記号"])#記号除く
                ans.append(surface)
            ans = " -> ".join(ans)#->で連結
            print(ans)
if __name__ == "__main__":
    fname = "ai.ja.txt.parsed"
    text = knock41.parse_chunk(fname)
    main(text[1:])#タイトルは除く