#!/usr/bin/python3


with open ('popular-names.txt') as f1:
    with open('11.txt', mode='w') as f2:
        for line in f1:
            tab2space = line.strip().replace("\t"," ")
            tab2space += "\n"
            f2.write(tab2space)