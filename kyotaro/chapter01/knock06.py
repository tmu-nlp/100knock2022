def n_gram(X, n):
    ans = []
    for i in range(len(X) - n + 1):
        ans.append(X[i:i+n])
    return ans

s = "paraparaparadise"
t = "paragraph"

X = set(n_gram(s, 2))
Y = set(n_gram(t, 2))

uni = X.union(Y)
inte = X.intersection(Y)
dif = X.difference(Y)
X_jud = 'se' in X
Y_jud = 'se' in Y

print("X:", X)
print("Y:", Y)
print("union:", uni)
print("intersection:", inte)
print("difference:", dif)
print("'se' in X?:", X_jud)
print("'se' in Y?:", Y_jud)