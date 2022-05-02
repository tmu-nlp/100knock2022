#!/usr/bin/python3

from collections import defaultdict #default-set

my_dict = defaultdict(lambda: 0)

with open ('popular-names.txt') as f1:
    trg_list = f1.readlines()
    for i in range(len(trg_list)):
        trg_list[i] = trg_list[i].replace('\n','')
        trg_list[i] = trg_list[i].split('\t')

for i in range(len(trg_list)):
    my_dict[trg_list[i][0]] += 1
# print(my_dict)

dict_sorted = sorted(my_dict.items(), key=lambda x:x[1], reverse = True)

print(dict_sorted)
