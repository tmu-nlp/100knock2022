a = open('./100knock2022/DUAN/chapter02/popular-names.txt').readlines()
b1 = open('./100knock2022/DUAN/chapter02/col1.txt','w')
b2 = open('./100knock2022/DUAN/chapter02/col2.txt','w')
for moji in a:
    b = moji.strip().split('\t')
    b1.writelines(b[0] + '\n')
    b2.writelines(b[1] + '\n')
