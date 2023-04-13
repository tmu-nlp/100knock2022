#100本ノック第1章09
#スペースで区切られた単語列に対して，各単語の先頭と末尾の文字は残し，それ以外の文字の順序をランダムに並び替えるプログラムを作成せよ．
# ただし，長さが４以下の単語は並び替えないこととする．適当な英語の文
# （例えば”I couldn’t believe that I could actually understand what I was reading : the phenomenal power of the human mind .”）
# を与え，その実行結果を確認せよ．

import random

#line = "I couldn’t believe that I could actually understand what I was reading : the phenomenal power of the human mind ."
line = input()
word_list = line.split()#単語は空白区切りでリストに格納
result = []

for i, word in enumerate(word_list):
    if len(word) > 4:#5文字以上の時
        char_list = list(word[1:-1])#単語の先頭と末尾を除く各文字をリストに格納
        random.shuffle(char_list)#リストをシャッフル

        result.append(word[0]+"".join(char_list)+word[-1])

    else:#4文字以下の時
        result.append(word)

print(" ".join(result))

