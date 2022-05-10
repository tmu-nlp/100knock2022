#knock37
#「猫」とよく共起する（共起頻度が高い）10語とその出現頻度をグラフ（例えば棒グラフなど）で表示せよ
import knock30, knock35
import matplotlib.pyplot as plt
import japanize_matplotlib#日本語対応してくれる

def get_surface(sentences):#単語の表層形だけのリストを作成
    surface_list = []
    for sentence in sentences:
        words = []
        for morpheme in sentence:
            words.append(morpheme["surface"])
        surface_list.append(words)
    return surface_list

def get_neko(surface_list):#猫を含む文で猫以外の単語を抽出
    words_list = list(filter(lambda x: '猫' in x, surface_list))#猫を含む文を抽出
    neko_list = []
    for words in words_list:
        for word in words:
            if word != "猫":#猫は含まない
                neko_list.append(word)
    return neko_list

if __name__ == "__main__":
    fname = "neko.txt.mecab"
    sentences = knock30.load_output(fname)#各形態素ごとのマップを1文ずつ保持したリスト
    surface_list = get_surface(sentences)#単語の表層形だけのリスト
    neko_list = get_neko(surface_list)#猫と共起する単語のリスト
    freq_dict = knock35.get_freq(neko_list)#key:単語，value:頻度の辞書
    result = {key[0]:freq_dict[key] for key in list(freq_dict)[:10]}#上位10こ
    #グラフ出力
    plt.xlabel("猫と共起する単語")
    plt.ylabel("頻度")
    plt.bar(result.keys(), result.values())#棒グラフ
    plt.savefig('output37.png')