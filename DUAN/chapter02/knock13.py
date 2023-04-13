#書き込みファイルと読み込みファイルを開く
a = open('./100knock2022/DUAN/chapter02/col.txt','w')
b1 = open('./100knock2022/DUAN/chapter02/col1.txt')
b2 = open('./100knock2022/DUAN/chapter02/col2.txt')

for moji1 in b1:
    moji2 = b2.readline()  #ファイル(col2.txt)を１行だけ読み込む
    a.write(moji1.strip() + '\t' + moji2.strip() + '\n') #タブ区切りで連結して出力する