# アルファベットのリストを作る
a = 'Now I need a drink, alcoholic of course, after the heavy lectures involving quantum mechanics.'

# replace()を使用して、','と'.'を空白に置換する
b = a.replace(',','').replace('.','')

# 空白ごとに単語を分割する
c = b.split()

# リストを作成して、各単語の文字数を取得し、出力する
pi = []
for i in c:
    pi.append(len(i))
print (pi)