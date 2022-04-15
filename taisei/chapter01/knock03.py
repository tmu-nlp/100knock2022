#003
# -*- coding: utf-8 -*-
str_info = 'Now I need a drink, alcoholic of course, after the heavy lectures involving quantum mechanics.'
count_list = []
word = ''
word_list = str_info.split(" ")
for i in range (len(word_list)):
    word = word_list[i]
    if (word[len(word) - 1] == ',' or word[len(word) - 1] == '.'): #単語の最終文字が',' か '.'の時はそれを省く
        word = word[0 : len(word) - 1]
        word_list[i] = word
    
for i in range (len(word_list)):
    count_list.append(len(word_list[i]))
print(word_list)
print(count_list)