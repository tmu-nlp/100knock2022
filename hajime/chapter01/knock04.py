#!/usr/bin/python3

from collections import defaultdict

trg_str = "Hi He Lied Because Boron Could Not Oxidize Fluorine. New Nations Might Also Sign Peace Security Clause. Arthur King Can."
my_list = trg_str.split(" ") 
list_index = [1] * len(my_list)
illegal = [1-1,5-1,6-1,7-1,8-1,9-1,15-1,16-1,19-1] 
for i in illegal:
    list_index[i] = 0
my_dict = defaultdict(lambda: 0)
#print(my_list)
#output = ""
for i in range(0,len(my_list)):
    sub = ""
    if list_index[i] == 0:
        sub = my_list[i][0] + my_list[i][1]
    elif list_index[i] == 1:
        sub = my_list[i][0]
    my_dict[sub] = i

print(my_dict)

#Hi H L B Bo Co No Ox Fl N N M A S P Se Ca A K Ca