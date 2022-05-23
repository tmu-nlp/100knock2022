#knock49 名詞間の係り受けパスの抽出
import knock41

def search_path(sentence, chunk, result):
    """ある文節スタートで根までパスをたどり，通ったノードをresultに格納する"""
    if chunk.dst == -1:#係先がなくなった時終わり
        return result
    result.append(sentence[chunk.dst])#係先の文節を追加
    return search_path(sentence, sentence[chunk.dst], result)#係先の文節にも同じ操作

def replace_noun_with_char(char, chunk):
    """名詞句の名詞をXかYに置換する"""
    surface = ""
    first_char = True
    for morph in chunk.morphs:
        if morph.pos == "名詞": 
            if first_char:
                surface += char
                first_char = False
        else:
            if morph.pos != "記号":
                surface += morph.surface
    return surface

def main(text):
    for sentence in text:
        #文中の名詞句ペアを探す
        #名詞句1
        for i in range(len(sentence)):
            noun_flag1 = False
            for morph in sentence[i].morphs:
                if morph.pos == "名詞":
                    noun_flag1 = True
            if not noun_flag1:
                continue
            #名詞句2
            for j in range(i + 1, len(sentence)):
                noun_flag2 = False
                for morph in sentence[j].morphs:
                    if morph.pos == "名詞":
                        noun_flag2 = True
                if not noun_flag2:
                    continue
                #各ペアについて以下の処理
                #1つ目の名詞句->根のパスを求める
                result1 = search_path(sentence, sentence[i], [sentence[i]])
                #パス上に2つ目の名詞句がある場合：48同様に連結して表示
                if sentence[j] in result1:
                    #名詞句1->名詞句2のパスを求める
                    ans = []
                    for chunk in result1:
                        ans.append(chunk)
                        if sentence[j] == chunk:
                            break
                    ans_chunk = []
                    #名詞句1はXに置換
                    surface = replace_noun_with_char("X",ans[0])
                    ans_chunk.append(surface)
                    #その他の文節を答えのリストに追加
                    for chunk in ans[1:len(ans) - 1]:
                        for morph in chunk.morphs:
                            surface = "".join([m.surface for m in chunk.morphs if m.pos != "記号"])#記号除く
                        ans_chunk.append(surface)
                    #名詞句2はYに置換
                    surface = replace_noun_with_char("Y",ans[-1])
                    ans_chunk.append(surface)
                    #->で連結して表示
                    ans_chunk = " -> ".join(ans_chunk)
                    print(ans_chunk)
                #パス上に2つ目がない場合
                else:
                    #2つ目の名詞句->根のパスを求める
                    result2 = search_path(sentence, sentence[j], [sentence[j]])
                    #名詞句1->根と名詞句2->根のパスで交差するノードを探す
                    #交差するノードのうち，最初に出てくるもの
                    cross_nodes = []
                    for chunk1 in result1:
                        for chunk2 in result2:
                            if chunk1 == chunk2:
                                cross_nodes.append(chunk1)
                    cross_node = cross_nodes[0]
                    #名詞句1はXに置換
                    ans_chunk1 = []
                    surface = replace_noun_with_char("X",result1[0])
                    ans_chunk1.append(surface)
                    #名詞句1->交差ノードまでのパスを保存
                    for chunk1 in result1[1:]:
                        if chunk1 == cross_node:
                            break
                        for morph in chunk1.morphs:
                            surface = "".join([m.surface for m in chunk1.morphs if m.pos != "記号"])#記号除く
                        ans_chunk1.append(surface)
                    #名詞句2はYに置換
                    ans_chunk2 = []
                    surface = replace_noun_with_char("Y",result2[0])
                    ans_chunk2.append(surface)
                    #名詞句2->交差ノードまでのパスを保存
                    for chunk2 in result2[1:]:
                        if chunk2 == cross_node:
                            break
                        for morph in chunk2.morphs:
                            surface = "".join([m.surface for m in chunk2.morphs if m.pos != "記号"])#記号除く
                        ans_chunk2.append(surface)
                    #交差ノードの表層形を保存
                    for morph in cross_node.morphs:
                        cross_surface = "".join([m.surface for m in cross_node.morphs if m.pos != "記号"])#記号除く
                    #|で連結して表示
                    ans_chunk1 = " -> ".join(ans_chunk1)
                    ans_chunk2 = " -> ".join(ans_chunk2)
                    print(f"{ans_chunk1} | {ans_chunk2} | {cross_surface}")
                    
if __name__ == "__main__":
    fname = "ai.ja.txt.parsed"
    text = knock41.parse_chunk(fname)
    main(text[1:11])#タイトルは除く，大きすぎるから10文まで
#交差ノードの出力がおかしい？->わかんない