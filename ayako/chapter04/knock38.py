#knock38
#単語の出現頻度のヒストグラムを描け．
# ただし，横軸は出現頻度を表し，1から単語の出現頻度の最大値までの線形目盛とする．
# 縦軸はx軸で示される出現頻度となった単語の異なり数（種類数）である．
import knock30, knock35
import matplotlib.pyplot as plt
import japanize_matplotlib#日本語対応してくれる

if __name__ == "__main__":
    fname = "neko.txt.mecab"
    sentences = knock30.load_output(fname)
    word_list = knock35.get_word(sentences)
    freq_dict = knock35.get_freq(word_list)
    #グラフ出力
    plt.xlabel("出現頻度")
    plt.ylabel("単語の異なり数")
    plt.hist(freq_dict.values(), range=(1,10))
    plt.savefig('output38.png')