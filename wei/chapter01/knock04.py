
res = dict()
s = 'Hi He Lied Because Boron Could Not Oxidize Fluorine. New Nations Might Also Sign Peace Security Clause. Arthur King Can.'
words = s.replace('.', '').split()
print(len(words))
indexs = [1, 5, 6, 7, 8, 9, 15, 16, 19]
for i,word in enumerate(words):
    if i+1 in indexs:
        char = word[0]
        res[char] = i+1

    else:
        chars = word[:2]
        res[chars] = i+1

print(res)