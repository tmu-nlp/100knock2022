# knock-40
# 形態素を表すクラスMorphを実装せよ．
# このクラスは表層形（surface），基本形（base），品詞（pos），品詞細分類1（pos1）をメンバ変数に持つこととする．
# さらに，係り受け解析の結果（ai.ja.txt.parsed）を読み込み，各文をMorphオブジェクトのリストとして表現し，冒頭の説明文の形態素列を表示せよ．

# 人工	名詞,一般,*,*,*,*,人工,ジンコウ,ジンコー
# 表層形\t品詞,品詞細分類1,品詞細分類2,品詞細分類3,活用型,活用形,原形,読み,発音
# https://taku910.github.io/cabocha/

class Morph():
    def __init__(self, morph):
        surface, attr = morph.split('\t')
        attr_list = attr.split(',')
        self.surface = surface  # 表層系
        self.base = attr_list[6]  # 原型
        self.pos = attr_list[0]  # 品詞
        self.pos1 = attr_list[1]  # 品詞細分類1


sentences = []
morphs = []

with open("ai.ja.txt.parsed", "r") as f:
    for line in f:
        if line[0] == '*':  # 開幕が*の場合は飛ばす
            continue
        elif line == "EOS\n":  # EOSとEOFを間違えない
            if len(morphs) > 0:  # morphsが1以上ある場合
                sentences.append(morphs)  # sentencesにmorphsのリストを追加
            morphs = []  # 初期化
        else:
            morphs.append(Morph(line))  # morphsにMorphクラスを追加

for mophes in sentences[1]:
    print(vars(mophes))  # 辞書型を一気に出力

# https://qiita.com/ganyariya/items/e01e0355c78e27c59d41
# varsでやると辞書型を一気に出力できる
