#004
# -*- coding: utf-8 -*-

def word_split(sen):
    word_list = sen.split(" ")
    for i in range (len(word_list)):
        word = word_list[i]
        if (word[len(word) - 1] == ',' or word[len(word) - 1] == '.'): #単語の最終文字が',' か '.'の時はそれを省く
            word = word[0 : len(word) - 1]
            word_list[i] = word
    return word_list

str_info2 = 'Hi He Lied Because Boron Could Not Oxidize Fluorine. New Nations Might Also Sign Peace Security Clause. Arthur King Can.'
word_list = word_split(str_info2)
first_letter_list = [1, 5, 6, 7, 8, 9, 15, 16, 19]
dic = {}

for i in range (1, len(word_list) + 1):
    if (i in first_letter_list):  #first_letter_listの数字は〇〇番目ってことを表すけど、k番目の単語はword_list[k-1]のことを指す
        dic[word_list[i - 1][0]] = i
        
    else:
        dic[word_list[i - 1][0] + word_list[i - 1][1]] = i 
print(dic)