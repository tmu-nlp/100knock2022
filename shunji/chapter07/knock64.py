"""
> 単語アナロジーの評価データをダウンロードし，vec(2列目の単語) - vec(1列目の単語) + vec(3列目の単語)を計算し，
そのベクトルと類似度が最も高い単語と，その類似度を求めよ．求めた単語と類似度は，各事例の末尾に追記せよ．
"""

import pickle

model = pickle.load(open("model.pkl", "rb"))

with open("questions-words.txt", "r") as rf:
    with open("./questions-words-add.txt", "w") as wf:

        # rfから1行ずつ読込み，求めた単語と類似度を追加してwfに書込む
        for line in rf:
            line = line.split()
            if line[0] == ":":
                category = line[1]
            else:
                word, cos = model.most_similar(
                    positive=[line[1], line[2]], negative=[line[0]], topn=1
                )[0]  # (単語, cos類似度)のリストが返ってくるため第一要素を取る
                
                wf.write(" ".join([category] + line + [word, str(cos) + "\n"]))
                print(" ".join([category] + line + [word, str(cos) + "\n"]))
