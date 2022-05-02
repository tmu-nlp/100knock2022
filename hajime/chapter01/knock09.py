#!/usr/bin/python3

import random

# 入力を受け取る
text = "I couldn’t believe that I could actually understand what I was reading : the phenomenal power of the human mind ."
# 空白で区切る
sentence = text.split(" ")
# print(sentence)

# 単語の個数だけfor文を回す
for i in range(len(sentence)):
    # i番目の単語の長さが4以下の場合は並び替えない
    if (len(sentence[i]) <= 4) :
        continue
    # 単語の長さが5以上の場合は並び替えを行う
    else:
        # 最初の文字と最後の文字だけ別途に抽出
        start = sentence[i][0]
        end = sentence[i][len(sentence[i])-1]
        # 中間の文字を切り出してrandom.sample関数でランダムに並び替える
        inter = sentence[i][1:len(sentence[i])-1]
        new_inter = ''.join(random.sample(inter,len(inter)))
        # 結合
        sentence[i] = start + new_inter + end

# list型は.joinを用いることで結合できる
new_sentence = ' '.join(sentence)
# 出力
print(new_sentence)

