import re
"""
*から始まる文の形式
1.*
2.文節番号
3.係り先の文節番号(係り先なし:-1)
4.主辞の形態素番号/機能語の形態素番号
5.係り関係のスコア(大きい方が係りやすい)
"""

class Chunk:
    def __init__(self, dst):
        self.morphs = [] #Morphオブジェクトのリスト
        self.dst = int(dst) #係先文節インデックス番号
        self.srcs = [] #係元文節インデックス番号のリスト


class Morph:
    def __init__(self, line):
        line = line.split('\t')
        self.surface = line[0]
        mor = line[1].split(',')
        self.base = mor[6]
        self.pos = mor[0]
        self.pos1 = mor[1]


morph_list = [] #1つの文のMorphオブジェクトのリスト
morph_list_list = [] #各文ごとのMorphオブジェクトのリストのリスト
chunk_list = [] #1つの文のChunkオブジェクトのリスト ai.ja.txt.parsedの文節(*と*の間にあるものをひとつの塊)が要素
chunk_list_list = [] #各文ごとのChunkオブフェクトのリストのリスト
with open("ai.ja.txt.parsed", "r") as f:
    for line in f:
        if re.match(r'\*', line):
            if re.match(r'\* 0 ', line):
                line = line.strip().split()
                chunk_obj = Chunk(re.sub(r'D', '', line[2]))
            else:
                chunk_list.append(chunk_obj)
                line = line.strip().split()
                chunk_obj = Chunk(re.sub(r'D', '', line[2])) #数字の後のDをとってあげる

        elif re.match(r'EOS$', line):
            if len(morph_list) > 0:
                chunk_list.append(chunk_obj)
                morph_list_list.append(morph_list)
                for i, chunk_obj in enumerate(chunk_list):
                    if chunk_obj.dst == -1:
                        continue
                    chunk_list[chunk_obj.dst].srcs.append(i)
                chunk_list_list.append(chunk_list)
                morph_list = []
                chunk_list = []

        else:
            morph_obj = Morph(line)
            morph_list.append(morph_obj)
            chunk_obj.morphs.append(morph_obj)

if __name__ == "__main__":
    with open("./output/knock41_output.txt", "w") as f_out:
        for k in chunk_list_list[1]:
            #f_out.write(f'{[morph.surface for morph in k.morphs]}  {k.dst}  {k.srcs}\n')
            f_out.write(f'{"".join([morph.surface for morph in k.morphs])}  {k.dst}  {k.srcs}\n')
