def n_gram(X, n):
    ans = []
    for i in range(len(X) - n + 1):
        ans.append(X[i:i+n])
    return ans

s = "I am an NLPer"

s_word = s.split()

print(n_gram(s_word, 2))
print(n_gram(s, 2))