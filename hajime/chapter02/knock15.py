#!/usr/bin/python3

import sys
num = sys.argv[1]
num = int(num)

with open ('popular-names.txt') as f1:
    trg_list = f1.readlines()
    trg_list_len = len(trg_list)
    for i in range(len(trg_list)):
        trg_list[i] = trg_list[i].replace('\n','')

for i in reversed(range(trg_list_len - num, trg_list_len)):
    print(trg_list[i])