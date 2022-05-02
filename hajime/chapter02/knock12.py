#!/usr/bin/python3

with open ('popular-names.txt') as f1:
    with open ('col1.txt','w') as f2:
        with open('col2.txt','w') as f3:
            for line in f1:
                trg_line = line
                trg_space = trg_line.strip().replace("\t"," ")
                trg_list = trg_space.split(" ") 
                f2.write(trg_list[0] + "\n")
                f3.write(trg_list[1] + "\n")
