a = 'パタトクカシーー'
b = []
for i in range(len(a)):
    if i%2 == 0:
        b.append(a[i])
print (''.join(b))
