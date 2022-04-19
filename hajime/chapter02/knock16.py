#!/usr/bin/python3

import sys
num = int(sys.argv[1])

with open ('popular-names.txt') as f1:
    trg_list = f1.readlines()

len_line = len(trg_list) // num

#print(len(trg_list))

for i in range(num):
    start = i * len_line
    text = trg_list[start:start+len_line]
    file_name = "sub-name-" + str(i) + ".txt"
    #print(text)
    #print(len(text))
    with open(file_name,'w') as f:
        for j in range(len(text)):
            f.writelines(text[j])

