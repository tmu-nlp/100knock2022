#!/usr/bin/python3

trg_string = "stressed"

# using slice [start:stop:step]
reverse1 = trg_string[::-1]
print(reverse1)

# using for range like c++
reverse2 = ""
# indexの逆順に出力
for i in range(len(trg_string)-1,-1,-1):
    reverse2 += trg_string[i]
print(reverse2)


