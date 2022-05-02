#!/usr/bin/python3

from collections import defaultdict

trg_sen = "Now I need a drink, alcoholic of course, after the heavy lectures involving quantum mechanics."
my_list = trg_sen.split(" ") 
#print(my_list)
trg_list = []
for i in range(0,len(my_list)):
    trg_list.append(len(my_list[i]))
print(trg_list)
