from dataclasses import replace
import re
import string

# 前処理
def pre_process(text):
    # replace punctuation marks with spaces, keeping equal length
    table =str.maketrans(
        string.punctuation, ' '*len(string.punctuation)
    )
    text = text.translate(table)
    text = text.lower()
    text = re.sub('[0-9]+', '0', text)

    return text

files_ja = [
    ["train.mecab.ja", "./tokenized/train.ja"],
    ["dev.mecab.ja", "./tokenized/dev.ja"],
    ["test.mecab.ja", "./tokenized/test.ja"]
]

for src, trg in files_ja:
    with open(src, 'r', encoding='utf-8') as f_in, open(trg, 'w', encoding='utf-8') as f_out:
        for i, line in enumerate(f_in):
            if i > 0:
                if line != 'EOS\n':
                    surface = line.split('\t')[0]
                    f_out.write(surface + ' ')
                else:
                    f_out.write('\n')


files_en = [
    ["train.en", "./tokenized/train.en"],
    ["dev.en", "./tokenized/dev.en"],
    ["test.en", "./tokenized/test.en"]
]

for src, trg in files_en:
    with open(src, 'r', encoding='utf-8') as f_in_en, open(trg, 'w', encoding='utf-8') as f_out_en:
        for line in f_in_en:
            f_out_en.write(pre_process(line))

