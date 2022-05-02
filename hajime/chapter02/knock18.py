#!/usr/bin/python3

with open ('popular-names.txt') as f1:
    trg_list = f1.readlines()
    for i in range(len(trg_list)):
        trg_list[i] = trg_list[i].replace('\n','')
        trg_list[i] = trg_list[i].split('\t')
# print(trg_list)

trg_list.sort(key=lambda x: x[2],reverse = True)
print(trg_list)
#途中 listを整形して出力する．
for i in range(len(trg_list)):
    output = trg_list[i][0] + "\t" + trg_list[i][1] + "\t" +trg_list[i][2] + "\t" +trg_list[i][3]
    print(output)