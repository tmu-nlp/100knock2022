a = 'パトカー'
b = 'タクシー'
c = []; i = 0
for i in range(len(a)):
    if i < 4:
        c.append(a[i]+b[i])
        i += 1
print(''.join(c))
