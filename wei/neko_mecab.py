
import MeCab
import ipadic


tagger = MeCab.Tagger(ipadic.MECAB_ARGS)
# txt ="今日はいい天気です"
# res = tagger.parse(txt)
# print(res)

f_path = "./data/neko.txt"
with open(f_path, 'r',encoding='utf-8') as f:
    for line in f:
        line = line.strip()
        res = tagger.parse(line)
        #res = res + '\n'
        with open('./data/neko.txt.mecab','a+', encoding='utf-8') as out_f:
            out_f.write(res)