a = 'Hi He Lied Because Boron Could Not Oxidize Fluorine. New Nations Might Also Sign Peace Security Clause. Arthur King Can.'
b = [0,4,5,6,7,8,14,15,18]
word = a.split()
ele = {}
for i, j in enumerate(word):
    if i in b:
        ele[i] = j[0]
    else:
        ele[i] = j[:2]
print (ele)