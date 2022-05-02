def n_gram(n, text):
    l = []
    if type(text) == str:
        for i in range(len(text) - n + 1):
            l.append(''.join(text[i:i+n]))
    else:
        for i in range(len(text) - n + 1):
            l.append(' '.join(text[i:i+n]))
    return l

text1 = 'paraparaparadise'
text2 = 'paragraph'

X = set(n_gram(2, text1))
Y = set(n_gram(2, text2))
print('X = ' + str(X))
print('Y = ' + str(Y))

union = X | Y
intersection = X & Y
diff_xy = X - Y
diff_yx = Y - X

print('X | Y = ' + str(union))
print('X & Y = ' + str(intersection))
print('X - Y = ' + str(diff_xy))
print('Y - X = ' + str(diff_yx))

bg = 'se'
if (bg in X):
    print(bg + ' is in X')
else:
    print(bg + ' is not in X')
if (bg in Y):
    print(bg + ' is in Y')
else:
    print(bg + ' is not in Y')
