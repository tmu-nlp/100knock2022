'''
30. 形態素解析結果の読み込み
各形態素は表層形(surface)、基本形(base)、品詞(pos)、品詞細分類(pos1)をキーとするマッピング型に格納し、
1文を形態素のリストとして表現せよ。

出力フォーマットは左から
表層形\t品詞,品詞細分類1,品詞細分類2,品詞細分類3,活用型,活用形,原形,読み,発音
となっています。
'''

def load_result(mecab_f):
    sentences = []
    list_of_dic = []
    with open(mecab_f, 'r', encoding='utf-8') as f:
        for line in f:
            if line != 'EOS\n':     # EOS\nは句の区切り
                dic = dict()        # {}にはこれから各単語の形態素を格納、len(dic)=4
                morphs = line.replace('\t', ',').split(',')
                dic['surface'] = morphs[0]
                dic['base'] = morphs[7]
                dic['pos'] = morphs[1]
                dic['pos1'] = morphs[2]
                list_of_dic.append(dic)    # len(sentence) == len(list_of_dic)

            else:
                sentences.append(list_of_dic)
                list_of_dic = []
    return list(filter(None, sentences))      # filter empty list

if __name__ == '__main__':
    file = '../data/neko.txt.mecab'
    results = load_result(file)
    print(len(results))               # 9210
    for morphs in results[:3]:
        # print(type(i))  # class dict
        print(morphs)
        #if len(morphs) == 1:

            #print(morphs)

'''[{'surface': '一', 'base': '一', 'pos': '名詞', 'pos1': '数'}]
[{'surface': '吾輩', 'base': '吾輩', 'pos': '名詞', 'pos1': '代名詞'}, {'surface': 'は', 'base': 'は', 'pos': '助詞', 'pos1': '係助詞'}, {'surface': '猫', 'base': '猫', 'pos': '名詞', 'pos1': '一般'}, {'surface': 'で', 'base': 'だ', 'pos': '助動詞', 'pos1': '*'}, {'surface': 'ある', 'base': 'ある', 'pos': '助動詞', 'pos1': '*'}, {'surface': '。', 'base': '。', 'pos': '記号', 'pos1': '句点'}]
[{'surface': '名前', 'base': '名前', 'pos': '名詞', 'pos1': '一般'}, {'surface': 'は', 'base': 'は', 'pos': '助詞', 'pos1': '係助詞'}, {'surface': 'まだ', 'base': 'まだ', 'pos': '副詞', 'pos1': '助詞類接続'}, {'surface': '無い', 'base': '無い', 'pos': '形容詞', 'pos1': '自立'}, {'surface': '。', 'base': '。', 'pos': '記号', 'pos1': '句点'}]'''