#準備 ターミナルで
#cabocha -f1 ai.ja.txt -o ai.ja.txt.parsed
#https://qiita.com/nezuq/items/f481f07fc0576b38e81d
import re

class Morph:
    def __init__(self, line):
        line = line.split('\t')
        self.surface = line[0]
        mor = line[1].split(',')
        self.base = mor[6]
        self.pos = mor[0]
        self.pos1 = mor[1]


morph_list = [] #各文ごとのMorphオブジェクトのリスト 
morph_list_list = [] #各文ごとのMorphオブジェクトのリストのリスト
with open("ai.ja.txt.parsed", "r") as f:
    for line in f:
        if re.match(r'\*', line):
            continue

        elif re.match(r'EOS$', line):
            if len(morph_list) > 0:
                morph_list_list.append(morph_list)
                morph_list = []

        else:
            morph_list.append(Morph(line))
            
if __name__ == "__main__":
    with open("./output/knock40_output.txt", "w") as f_out:
        for morph in morph_list_list[1]:
            f_out.write(f'{morph.__dict__}\n')
