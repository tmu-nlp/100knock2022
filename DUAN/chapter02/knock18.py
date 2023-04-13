a = open('./100knock2022/DUAN/chapter02/popular-names.txt').readlines()
a.sort(key=lambda b: int(b.split('\t')[2]), reverse=True)
for moji in a:
    print(moji.strip())