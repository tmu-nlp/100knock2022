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
    rs_eos = re.match(r'EOS', line)
    if rs:
        morpheme_dict["表層形"] = rs.group(1)
        morpheme_dict["基本形"] = rs.group(8)
        morpheme_dict["品詞"] = rs.group(2)
        morpheme_dict["品詞細分類1"] = rs.group(3)
        morpheme_line.append(morpheme_dict)
        morpheme_dict = {}

    elif rs_eos:
        morpheme_line_list.append(morpheme_line)
        morpheme_line = []

with open("knock30_output.txt", "w") as f_out:
    for x in morpheme_line_list:
        f_out.write(f'{x}\n')