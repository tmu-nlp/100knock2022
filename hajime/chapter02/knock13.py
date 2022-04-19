#!/usr/bin/python3

with open ('concat.txt','w') as f1:
    with open ('col1.txt','r') as f2:
        with open('col2.txt','r') as f3:
            list1 = f2.readlines()
            list2 = f3.readlines()
            for i in range(len(list1)):
                list1[i] = list1[i].replace('\n','')
                list2[i] = list2[i].replace('\n','')
                trg_line = list1[i] + "\t" + list2[i] + "\n"
                f1.write(trg_line)


