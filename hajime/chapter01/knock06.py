#!/usr/bin/python3

def n_gram_char(seq,num):
    ngramset = list()
    for i in range(len(seq)-num+1):
        ngramset.append(seq[i:i+num])
    ngramset = list(set(ngramset))
    ngramset.sort()
    return ngramset

X = "paraparaparadise"
Y = "paragraph"

Xset = set(n_gram_char(X,2))
Yset = set(n_gram_char(Y,2))

union = Xset | Yset
print(union)
inter = Xset & Yset
print(inter)
XdifY = Xset - Yset
print(XdifY)
YdifX = Yset - Xset
print(YdifX)

if 'se' in X:
    print("'se' in X")
    
if 'se' in Y:
    print("'se' in Y")