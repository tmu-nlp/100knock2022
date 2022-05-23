## knock42. 係り元と掛かり先の文節の表示

#係り元の文節と係り先の文節のテキストをタブ区切り形式ですべて抽出せよ．
# ただし，句読点などの記号は出力しないようにせよ．

from knock41 import sentences #解析結果の読み込み

sentence = sentences[1] #1行目の解析

for chunk in sentence.chunks:
    if chunk.dst != -1: #掛かり先がある

        modifiers = [] #係り先(する方)の表層系を格納
        modifiees = [] #係り元(される方)の表層系を格納

        #掛かり先
        for morph in chunk.morphs:
            if morph.pos != '記号':
                modifier = morph.surface
                modifiers.append(modifier)

        #係り元
        for morph in sentence.chunks[chunk.dst].morphs: #掛かり先番号を指定して、係り元の情報を取り出す
            if morph.pos != '記号':
                modifiee = morph.surface
                modifiees.append(modifiee)  

        if __name__ == '__main__':
            #掛かり先文節内の文字を連結して表示
            print(''.join(modifiers), ''.join(modifiees), sep='\t')

            #形態素と掛かり先関係を見やすくする
            #print('+'.join(modifiers), '+'.join(modifiees), sep=' --> ')
