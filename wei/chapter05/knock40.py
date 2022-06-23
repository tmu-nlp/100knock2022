'''
40. 係り受け解析結果の読み込み(形態素)
形態素を表すクラスMorphを実装．
このクラスは表層形（surface），基本形（base），品詞（pos），品詞細分類1（pos1）をメンバ変数に持つこと
ai.ja.txt.parsedを読み込み，各文をMorphオブジェクトのリストとして表現し，冒頭の説明文の形態素列を表示

ai.ja.txt.parsedの中に、形態素解析出力フォーマット：タブやカンマで区切られいる。また形態素解析済みデータについて、文節の区切り情報が付与される
アスタリスク(⋆記号)：文節の開始位置
Morph　object: ファイルの各行を指定のフォーマットに変換
vars([object])関数(組み込み)：クラスの属性をdict型で返す
'''

class Morph:
    def __init__(self, line):
        surface, morphs = line.strip().split('\t')
        morphs = morphs.split(',')
        self.surface = surface
        self.base = morphs[6]
        self.pos = morphs[0]
        self.pos1 = morphs[1]


def get_morphs(file):
    with open(file, 'r', encoding='utf-8') as f:
        sentences = []
        morphs = []
        for line in f:
            if line[0] == '*':
                continue
            elif line != 'EOS\n':
                morphs.append(Morph(line))   # obtain morphs
            else:   # EOS(文末)の場合
                sentences.append(morphs)    #文ごとにリストに保存
                morphs = []     # 前の句の内容をクリア


    return sentences

# 出力結果を確認
if __name__ == '__main__':
    file_path = '../data/ai.ja.txt.parsed'
    sentences = get_morphs(file_path)
    sentences = list(filter(None, sentences))
    for i in sentences[1]:

        print(vars(i))
'''
{'surface': '人工', 'base': '人工', 'pos': '名詞', 'pos1': '一般'}
{'surface': '知能', 'base': '知能', 'pos': '名詞', 'pos1': '一般'}'''