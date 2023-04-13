#100本ノック第1章06
#“paraparaparadise”と”paragraph”に含まれる文字bi-gramの集合を，それぞれ, XとYとして求め，
# XとYの和集合，積集合，差集合を求めよ．さらに，’se’というbi-gramがXおよびYに含まれるかどうかを調べよ．

#セットっていうライブラリ？

def char_ngram(line,n):
    result_char = []

    for i in range(len(line)-n+1):#文字bi-gram
        result_char.append(line[i:i+n])

    return(result_char)

X = char_ngram("paraparaparadise",2)
Y = char_ngram("paragraph",2)

#set型を使えば集合演算が可能らしい
setX = set(X)
setY = set(Y)

#X,Yの和集合
set_or = setX | setY
print("X,Yの和集合：",set_or)

#X,Yの積集合
set_and = setX & setY
print("X,Yの積集合：",set_and)

#X,Yの差集合
set_sub1 = setX - setY
print("X-Yの差集合：",set_sub1)
set_sub2 = setY - setX
print("Y-Xの差集合：",set_sub2)

#’se’というbi-gramがXおよびYに含まれるかどうか
set_se = {"se"}
print("Xに含まれるか：",set_se <= setX)
print("Yに含まれるか：",set_se <= setY)
