#!/usr/bin/python3

concat_str = "パタトクカシーー"

trg_string1 = concat_str[0] + concat_str[2] + concat_str[4] + concat_str[6]
print(trg_string1)

trg_string2 = ""
for i in range(0,len(concat_str),2):
    trg_string2 += concat_str[i]
print(trg_string2)