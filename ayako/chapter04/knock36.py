#knock36
#出現頻度が高い10語とその出現頻度をグラフ（例えば棒グラフなど）で表示せよ．
import knock30, knock35
import matplotlib.pyplot as plt
import japanize_matplotlib#日本語対応してくれる

if __name__ == "__main__":
    fname = "neko.txt.mecab"
    sentences = knock30.load_output(fname)
    word_list = knock35.get_word(sentences)
    freq_dict = knock35.get_freq(word_list)
    result = {key[0]:freq_dict[key] for key in list(freq_dict)[:10]}#上位10こ
    #グラフ出力
    labels = [key for key in result.keys()]
    values = [value for key, value in result.items()]
    plt.xlabel("単語")
    plt.ylabel("頻度")
    plt.bar(labels, values)
    plt.savefig('output36.png')