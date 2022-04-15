#!/usr/bin/python3

import random

text = "I couldn’t believe that I could actually understand what I was reading : the phenomenal power of the human mind ."
sentence = text.split(" ")

for i in range(len(sentence)):
    if (len(sentence[i]) <= 4) :
        continue
    else:
        start = sentence[i][0]
        end = sentence[i][len(sentence[i])-1]
        inter = sentence[i][1:len(sentence[i])-1]
        new_inter = ''.join(random.sample(inter,len(inter)))
        sentence[i] = start + new_inter + end

new_sentence = ' '.join(sentence)
print(new_sentence)

