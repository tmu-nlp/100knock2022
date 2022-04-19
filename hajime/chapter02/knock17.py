#!/usr/bin/python3

with open ('popular-names.txt') as f1:
    trg_list = f1.readlines()
    for i in range(len(trg_list)):
        trg_list[i] = trg_list[i].replace('\n','')

str_type = set()
trg_str1 = trg_list[0].split(" ")
trg_str2 = trg_str1[0].split("\t")
for item in trg_str2:
    str_type.add(item)
print(str_type)
