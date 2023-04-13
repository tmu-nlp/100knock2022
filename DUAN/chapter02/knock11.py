a = open('./100knock2022/DUAN/chapter02/popular-names.txt').readlines()
for moji in a:
    print(moji.strip().replace('\t',' '))