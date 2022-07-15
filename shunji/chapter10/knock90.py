"""データをmecabに通して形態素解析した出力を整形する"""
filenames = [
    ["train.mecab.ja", "train.ja"],
    ["dev.mecab.ja", "dev.ja"],
    ["test.mecab.ja", "test.ja"],
]
for src, dst in filenames:
    with open(src, "r") as rf, open(dst, "w") as wf:
        for i, line in enumerate(rf):
            if i > 1:
                if line != "EOS\n":
                    surface = line.split("\t")[0]
                    wf.write(surface + " ")
                else:
                    wf.write("\n")

# enは./kftt-data-1.0/data/tok　以下のファイルを使用