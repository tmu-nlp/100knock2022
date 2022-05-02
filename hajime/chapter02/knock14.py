#!/usr/bin/python3

import sys
num = sys.argv[1]
num = int(num)

with open ('popular-names.txt') as f1:
    trg_list = f1.readlines()
    for i in range(len(trg_list)):
        trg_list[i] = trg_list[i].replace('\n','')

for i in range(num):
    print(trg_list[i])