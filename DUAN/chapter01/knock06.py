# 文字のbi-gramを返す
def bigram(n, str):
    a = []
# enumerateを使用して、インデックスと要素を同時に取得する
    for i, j in enumerate(str):
# 2文字毎の文字列に分割して、aに文字列を追加する
        if i + n > len(str):
            return a
        a.append(str[i:i+n])
        
str1 = 'paraparaparadise'
str2 = 'paragraph'

# set()を使用して、集合に変換する
X = set(bigram(2, str1))
Y = set(bigram(2, str2))
print(X)
print(Y)

# XとYの和集合，積集合，差集合
print(X|Y)
print(X&Y)
print(X-Y)

# in演算子を用いて、要素の判定を行う。
print('se' in X)
print('se' in Y)