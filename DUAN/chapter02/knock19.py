lines = open('./100knock2022/DUAN/chapter02/popular-names.txt').readlines()
col1_count = dict()
for l in lines:
    c = l.split("\t")[0]
    if col1_count.get(c):
        col1_count[c]+=1
    else:
        col1_count[c] = 1
sorted_col1 = sorted(col1_count.items(), key=lambda x: x[1], reverse=True)
print(*sorted_col1, sep="\n")