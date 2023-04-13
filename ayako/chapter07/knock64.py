# knock64
# 単語アナロジーの評価データをダウンロードし，
# vec(2列目の単語) - vec(1列目の単語) + vec(3列目の単語)を計算し，
# そのベクトルと類似度が最も高い単語と，その類似度を求めよ．
# 求めた単語と類似度は，各事例の末尾に追記せよ．
from gensim.models import keyedvectors

if __name__ == "__main__":
    model = keyedvectors.load_word2vec_format("GoogleNews-vectors-negative300.bin.gz", binary=True)
    with open("questions-words.txt", "r") as input_f, open("output/output64.txt", "w") as output_f:
        for line in input_f:
            words = line.split()#列は空白区切り
            if len(words) == 4:#カテゴリ名の行以外
                ans_vec = model.most_similar(positive=[words[1],words[2]], negative=[words[0]], topn=1)[0]#要素1個のリストで帰ってくるから[0]指定
                #ans_vec = [単語, 類似度]
                words.append(f"{ans_vec[0]} {ans_vec[1]}")
                print(" ".join(words), file=output_f)
            else:#カテゴリ名の行はそのまま出力
                print(line.strip(), file=output_f)