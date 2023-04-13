#knock39
#単語の出現頻度順位を横軸，その出現頻度を縦軸として，両対数グラフをプロットせよ．
import knock30, knock35
import matplotlib.pyplot as plt
import japanize_matplotlib#日本語対応してくれる

if __name__ == "__main__":
    fname = "neko.txt.mecab"
    sentences = knock30.load_output(fname)
    word_list = knock35.get_word(sentences)
    freq_dict = knock35.get_freq(word_list)
    #グラフ出力
    plt.xlabel("出現頻度順位")
    plt.ylabel("出現頻度")
    plt.xscale('log')
    plt.yscale('log')
    plt.scatter(range(1,len(freq_dict)+1), freq_dict.values())
    plt.savefig('output39.png')