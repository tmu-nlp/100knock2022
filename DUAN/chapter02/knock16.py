import math

n = int(sys.argv[1]) # nを取得する
lines = open('./100knock2022/DUAN/chapter02/popular-names.txt').readlines() # ファイルを読み込む
count = len(lines) # 長さを計算する
unit = math.ceil(count / n)  # 一つのファイルの行数

for i, j in enumerate(range(0, count, unit), 1): # インデックスと要素を同時に取り出す
    with open('out_{:02d}.txt'.format(i), mode='w') as out: # 書式を指定して、出力ファイルを作る
        for line in lines[j:j + unit]:
            out.write(line) # ファイルへの出力
