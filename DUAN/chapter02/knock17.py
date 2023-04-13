a =  open('./100knock2022/DUAN/chapter02/popular-names.txt')
set = set()
for line in a:
    set.add(line.split('\t')[0])
for moji in set:
    print(moji)