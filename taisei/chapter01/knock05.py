#005
# -*- coding: utf-8 -*-
def ngram_word(n, l):
    ngram_list = []
    for i in range (len(l) - n + 1):
        str = ''
        for j in range (n):
            str = str + " " + l[i + j] 
        
        ngram_list.append(str[1:len(str)]) #str[0] はスペースだから除いてる
    return ngram_list

def ngram_letter(n, sen):
    ngram_list = []
    for i in range (len(sen) - n + 1):
        str = ''
        for j in range (n):
            str = str + sen[i + j]
        ngram_list.append(str)
    return ngram_list

def word_split(sen):
    word_list = sen.split(" ")
    for i in range (len(word_list)):
        word = word_list[i]
        if (word[len(word) - 1] == ',' or word[len(word) - 1] == '.'): #単語の最終文字が',' か '.'の時はそれを省く
            word = word[0 : len(word) - 1]
            word_list[i] = word
    return word_list

print(ngram_word(2, word_split('I am an NLPer')))

print(ngram_letter(2, 'I am an NLPer'))