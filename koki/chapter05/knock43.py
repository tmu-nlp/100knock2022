## knock43. 名詞を含む文節が動詞を含む文節に係るものを抽出

#名詞を含む文節が，動詞を含む文節に係るとき，これらをタブ区切り形式で抽出せよ．
# ただし，句読点などの記号は出力しないようにせよ．

#knock42に名詞、動詞を含むかの判定を加筆するだけ

from knock41 import sentences #解析結果の読み込み

sentence = sentences[1] #1行目の解析

for chunk in sentence.chunks:
    if (chunk.dst != -1): #掛かり先がある

        modifiers = [] #係り先(する方)の表層系を格納
        modifiees = [] #係り元(される方)の表層系を格納

        flag_noun = 0 #名詞あるかないか
        flag_verb = 0 #動詞

        #掛かり先
        for morph in chunk.morphs:
            if morph.pos != '記号':
                modifier = morph.surface
                modifiers.append(modifier)     
            if morph.pos == '名詞':
                flag_noun = 1

        #係り元
        for morph in sentence.chunks[chunk.dst].morphs: #掛かり先番号を指定して、係り元の情報を取り出す
            if morph.pos != '記号':
                modifiee = morph.surface
                modifiees.append(modifiee)  
            if morph.pos == '動詞':
                flag_verb = 1
        
        if (flag_noun and flag_verb) == 1:
            print(''.join(modifiers), ''.join(modifiees), sep='\t')

