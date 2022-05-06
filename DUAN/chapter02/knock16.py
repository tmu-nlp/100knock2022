import math

n = 5 # Nを取得する

# ファイルの読み込み
lines = open('./100knock2022/DUAN/chapter02/popular-names.txt').readlines()

count = len(lines)
unit = math.ceil(count / n)  # 1ファイル当たりの行数

for i, j in enumerate(range(0, count, unit), 1):
    with open('child_{:02d}.txt'.format(i), mode='w') as out_file:
        for line in lines[j:j + unit]:
            out_file.write(line)