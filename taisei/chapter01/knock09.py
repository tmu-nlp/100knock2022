#009 
# -*- coding: utf-8 -*-
import random
def word_split(sen):
    word_list = sen.split(" ")
    for i in range (len(word_list)):
        word = word_list[i]
        if (word[len(word) - 1] == ',' or word[len(word) - 1] == '.'): #単語の最終文字が',' か '.'の時はそれを省く
            word = word[0 : len(word) - 1]
            word_list[i] = word
    return word_list

word_list = word_split('I couldn’t believe that I could actually understand what I was reading : the phenomenal power of the human mind .')
sentence_new = ''

for i in range (len(word_list)):
    word = word_list[i]
    letter_list = list(word) #文字をswapできるために単語をリスト化(wordを1文字ごとのリストに)。str内ではswapできないみたい
    if (len(word) <= 4):
        pass
    else:
        for _ in range (10): #10回swap
            a, b = random.randint(1, len(word) - 2), random.randint(1, len(word) - 2)
            tmp = letter_list[a]                        
            letter_list[a] = letter_list[b]
            letter_list[b] = tmp
        str = ''
        for k in letter_list:
            str += k
        word_list[i] = str
    sentence_new += word_list[i] + " "
sentence_new += '.'

print(sentence_new)