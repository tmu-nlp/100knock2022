def letter_n_gram(target, n):#文字n-gram
    result = []
    for i in range(len(target) - n + 1):
        result.append(target[i:i+n])
    return result

str1 = 'paraparaparadise'
str2 = 'paragraph'

X = letter_n_gram(str1, 2)
Y = letter_n_gram(str2, 2)

setX, setY = set(X), set(Y)
print('setX:',setX)
print('setY : ',setY)

#和集合
union = setX | setY
print('union : ',union)

#積集合
intersec = setX & setY
print('intersection : ',intersec)

#差集合
diff = setX - setY
print('difference : ',diff)

#内在判定
if 'se' in X:
    print('se in X')
if 'se' in Y:
    print('se in Y')
