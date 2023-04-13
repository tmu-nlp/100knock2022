from dataclasses import replace
import re
import string


def preprocessing(text):
    """記号除去，小文字化，数字を統一する前処理関数"""
    table = str.maketrans(
        string.punctuation, " " * len(string.punctuation)
    )  # string.punctuationは記号の文字列，そのそれぞれの記号をキー，スペースを値とした辞書を作成
    text = text.translate(table)  # tableにあるキーに該当したらそのキーに対応する値(スペース)に変換する．
    text = text.lower()  # 小文字化
    text = re.sub("[0-9]+", "0", text)  # 数字列を0に統一
    # text = re.sub(r'\s+?\n', ' ', text)

    return text


filenames_ja = [
    ["train.mecab.ja", "./tokenized_data/train.ja"],
    ["dev.mecab.ja", "./tokenized_data/dev.ja"],
    ["test.mecab.ja", "./tokenized_data/test.ja"],
]
for src, dst in filenames_ja:
    with open(src, "r") as rf, open(dst, "w") as wf:
        for i, line in enumerate(rf):
            if i > 1:
                if line != "EOS\n":
                    surface = line.split("\t")[0]
                    wf.write(surface + " ")
                else:
                    wf.write("\n")

filenames_en = [
    ["train.en", "./tokenized_data/train.en"],
    ["dev.en", "./tokenized_data/dev.en"],
    ["test.en", "./tokenized_data/test.en"],
]
for src, dst in filenames_en:
    with open(src, "r") as rf, open(dst, "w") as wf:
        for line in rf:
            wf.write(preprocessing(line))
