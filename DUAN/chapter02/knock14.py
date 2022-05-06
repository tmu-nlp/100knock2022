N = int(input())
with open("./100knock2022/DUAN/chapter02/popular-names.txt") as f:
    lines = f.readlines()
    print(*lines[:N], sep="")