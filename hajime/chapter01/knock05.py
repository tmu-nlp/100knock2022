#!/usr/bin/python3

def n_gram_word(seq,num):
    ngramset = list()
    for i in range(len(seq)-num+1):
        source = list()
        for j in range(i,i+num):
            source.append(seq[j])
        ngramset.append(source)
    return ngramset

def n_gram_char(seq,num):
    ngramset = list()
    for i in range(len(seq)-num+1):
        ngramset.append(seq[i:i+num])
    ngramset = list(set(ngramset))
    ngramset.sort()
    return ngramset

test = "I am an NLPer"
sentence = test.split(" ")
print(n_gram_word(sentence,2))
print(n_gram_char(test,2))

#重複の削除
