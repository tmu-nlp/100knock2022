import re
#http://taku910.github.io/mecab/#parse
with open("neko.txt.mecab", "r") as f:
    data = f.readlines()

morpheme_dict = {}
morpheme_line = []
morpheme_line_list = []

for line in data: 
    #line = line.strip()をすると文頭のスペース（段落）を消しちゃうから今回はしない
    rs = re.match(r'(.*?)\t(.*?)\,(.*?)\,(.*?)\,(.*?)\,(.*?)\,(.*?)\,(.*?)\,(.*?)\,(.*?)', line)
    rs2 = re.match(r'(.*?)\t(.*?)\,(.*?)\,(.*?)\,(.*?)\,(.*?)\,(.*?)', line)
    rs_eos = re.match(r'EOS', line)
    """
    これだとマッチしないものもあることに気づいた（e.g. ぷうぷうと）
    if rs:
        morpheme_dict["表層形"] = rs.group(1)
        morpheme_dict["基本形"] = rs.group(8)
        morpheme_dict["品詞"] = rs.group(2)
        morpheme_dict["品詞細分類1"] = rs.group(3)
        morpheme_line.append(morpheme_dict)
        morpheme_dict = {}
    """
    if rs2:

        line = line.split('\t')
        morpheme_dict["表層形"] = line[0]
        mor = line[1].split(',')
        morpheme_dict["基本形"] = mor[6]
        morpheme_dict["品詞"] = mor[0]
        morpheme_dict["品詞細い分類1"] = mor[1]
        morpheme_line.append(morpheme_dict)
        morpheme_dict = {}

    elif rs_eos:
        morpheme_line_list.append(morpheme_line)
        morpheme_line = []

with open("./output/knock30_output.txt", "w") as f_out:
    for x in morpheme_line_list:
        f_out.write(f'{x}\n')