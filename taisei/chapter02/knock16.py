import sys
f = open('popular-names.txt', 'r')
data = f.readlines()
n = int(sys.argv[1])
remainder = len(data) % n #dataの行数を分割ファイル数nで割ったあまり。
num_line = len(data) // n #先頭remainder個のファイルは行数がnum_line + 1。後ろn - remainder個のファイルは行数がnum_lineとなるようにする
count = 0
for i, line in enumerate(data):
    line = line.strip().split('\t')
    data[i] = line

for i in range(n):
    f_split = open('knock16_output_' + str(i) + '.txt', 'w')
    for _ in range (num_line):
        f_split.write('\t'.join(data[count]) + '\n')
        count += 1

    if (remainder != 0):
        f_split.write('\t'.join(data[count]) + '\n')
        count += 1
        remainder -= 1
    f_split.close()
f.close()
