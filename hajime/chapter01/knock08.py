#!/usr/bin/python3

import sys

def cipher(trg_str):
    output = ""
    for i in range(len(trg_str)):
        if trg_str[i].islower():
            small = ord(trg_str[i])
            output += chr(219 - small)
    return output
            
trg_str = sys.argv[1]

print(cipher(trg_str))