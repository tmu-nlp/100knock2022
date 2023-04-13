#knock35
#文章中に出現する単語とその出現頻度を求め，出現頻度の高い順に並べよ．
import knock30
import pandas as pd

def get_word(sentences):
    word_list = []
    for sentence in sentences:
        for morpheme in sentence:
            word_list.append(morpheme["surface"])
    return word_list

def get_freq(word_list):
    df = pd.DataFrame(word_list)#データフレームに変換
    freq_dict = df.value_counts().to_dict()#ユニークな要素とその出現回数の辞書を作成
    #単語がタプルで返って来ちゃうから
    return freq_dict

if __name__ == "__main__":
    fname = "neko.txt.mecab"
    sentences = knock30.load_output(fname)
    word_list = get_word(sentences)
    freq_dict = get_freq(word_list)
    for key, value in freq_dict.items():
        print(key[0],"\t", value)
