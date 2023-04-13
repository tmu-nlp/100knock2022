#knock30
#形態素解析結果（neko.txt.mecab）を読み込むプログラムを実装せよ．
#ただし，各形態素は表層形（surface），基本形（base），品詞（pos），品詞細分類1（pos1）をキーとするマッピング型に格納し，
#1文を形態素（マッピング型）のリストとして表現せよ．第4章の残りの問題では，ここで作ったプログラムを活用せよ．

#mecabの出力フォーマットは表層形\t品詞,品詞細分類1,品詞細分類2,品詞細分類3,活用型,活用形,原形,読み,発音
def load_output(fname):
    out_file = open(fname, "r")
    sentence = []#一文
    sentences = []#全文
    for line in out_file.readlines():
        line = line.strip("EOS\n").split("\t")#まずタブで区切る
        if line[0] == "":
            continue
        if len(line) > 1:
            new_line = line[1].split(",")
            #辞書を作成して格納
            morpheme = {
                "surface" : line[0],#表層形
                "base" : new_line[6],#原形
                "pos" : new_line[0],#品詞
                "pos1" : new_line[1]#品詞細分類1
            }
            sentence.append(morpheme)
            #句点来たら文終わり
            if morpheme["pos1"] == "句点":
                sentences.append(sentence)
                sentence = []
    return sentences

if __name__ == "__main__":
    fname = "neko.txt.mecab"
    sentences = load_output(fname)
    for sentence in sentences:
        print(sentence)
