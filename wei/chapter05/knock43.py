'''
43. 名詞を含む文節が動詞を含む文節に係るものを抽出
名詞を含む文節が、動詞を含む文節に係るとき、これらをタブ区切り形式で抽出
その際、句読点等の記号は出力しない
'''

from knock41 import *

if __name__ == '__main__':
    file_path = '../data/ai.ja.txt.parsed'
    sentences = get_chunks(file_path)
    for chunk in sentences[1].chunks:
        if int(chunk.dst) == -1:
            continue
        else:
            # 係り元文節
            modifier = ''.join([morph.surface for morph in chunk.morphs if morph.pos != '記号'])
            pos_noun = [morph.surface for morph in chunk.morphs if morph.pos == '名詞']
            # 係り先文節
            modifiee = ''.join([morph.surface for morph in sentences[1].chunks[int(chunk.dst)].morphs if morph.pos != '記号'])
            pos_verb = [morph.surface for morph in sentences[1].chunks[int(chunk.dst)].morphs if morph.pos == '動詞']
            if pos_noun and pos_verb:
                print(f'{modifier}\t{modifiee}')