#!/usr/bin/python3

import sys
import math
import numpy as np
num = int(sys.argv[1])

with open ('popular-names.txt') as f1:
    trg_list = f1.readlines()

len_line = len(trg_list) // num
rest = len(trg_list) % num

# print(trg_list)
ans = list(np.array_split(trg_list,num))
# print(ans[0])

for i in range(num):
    file_name = "sub-name-" + str(i) + ".txt"
    with open(file_name,'w') as f:
        for j in range(len(ans[i])):
            f.writelines(ans[i][j])

# for i in range(num):
#     start = i * len_line
#     text = trg_list[start:start+len_line]
#     file_name = "sub-name-" + str(i) + ".txt"
#     #print(text)
#     #print(len(text))
#     with open(file_name,'w') as f:
#         for j in range(len(text)):
#             f.writelines(text[j])

