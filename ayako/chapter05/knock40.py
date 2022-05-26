#knock40
#係受け解析結果の読み込み（形態素）

class Morph:
    def __init__(self, morpheme):
        self.surface = morpheme["surface"]
        self.base = morpheme["base"]
        self.pos = morpheme["pos"]
        self.pos1 = morpheme["pos1"]

def load_file(fname):
    text = input_file.read().split("EOS\n")#EOSで区切って配列格納
    text = list(filter(lambda x: x != "",text))#空白行削除
    return text

def parse_cabocha(sentence):
    ans = []
    for line in sentence.split("\n"):
        if line == '':
            return ans
        elif line[0] == "*":#ここで文節区切れる
            continue
        line = line.split("\t")#タブ区切りでsurfaceとその他を分割
        new_line = line[1].split(",")
        morpheme = {
            "surface":line[0],
            "base":new_line[6],
            "pos":new_line[0],
            "pos1":new_line[1]
        }
        ans.append(Morph(morpheme))
    return ans

if __name__ == "__main__":
    with open("ai.ja.txt.parsed", "r") as input_file:
        text = load_file(input_file)
        morphemes = [parse_cabocha(sentence) for sentence in text]
        for morpheme in morphemes[1]:#冒頭の説明文のみ
           print(vars(morpheme))#オブジェクトを辞書で返す