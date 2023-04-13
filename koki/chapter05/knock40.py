class Morph():
    def __init__(self, morph):
        surface, info = morph.split('\t') #\t区切りの表層形とそれ以外を切り分ける
        info = info.split(',') #info = [品詞,品詞細分類1,品詞細分類2,品詞細分類3,活用型,活用形,原形,読み,発音]
        self.surface = surface #表層形
        self.base = info[6] #基本形
        self.pos = info[0] #品詞
        self.pos1 = info[1] #品詞細分類1

morphs = [] #形態素ごと解析結果のオブジェクト配列
sentences = [] #1フレーズごとに解析結果を管理

with open('ai.ja.txt.parsed') as f:
    for line in f:
        if line[0] == '*': #係り受け解析結果は無視
            continue
        elif line == 'EOS\n': #EOSも無視(\nがないとだめだった)
            if len(morphs) > 0: #連続してEOSが続く場合など、空文字がsentencesに格納されることを防ぐ
                sentences.append(morphs)
                morphs = []
        else:
            morph_result = Morph(line) #インスタンス化
            morphs.append(morph_result)

for line in sentences[1]:
    #print(line.__dict__)
    print(vars(line)) #iのすべてのインスタンス情報を辞書で取得する、var()も同様
    #print('surface : {}\tbase : {}\tpos : {}\tpos1 : {}'.format(line.surface, line.base, line.pos, line.pos1)) 
