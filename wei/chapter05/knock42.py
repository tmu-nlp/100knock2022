'''
50. 係り元と係り先の文節を表示
係り元の文節と係り先の文節のテキストをタブ区切り形式ですべて抽出せよ．
ただし，句読点などの記号は出力しないようにせよ．
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
            # 係り先文節は係り先番号により取得
            modifiee = ''.join([morph.surface for morph in sentences[1].chunks[int(chunk.dst)].morphs if morph.pos != '記号'])
            print(f'係り元文節:{modifier}\t係り先文節:{modifiee}')

