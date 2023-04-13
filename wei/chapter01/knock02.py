'''
「パトカー」＋「タクシー」＝「パタトクカシーー」
'''
a = 'パトカー'
b = 'タクシー'
res = []
for i in range(len(a)):
    res.append(a[i])
    res.append(b[i])

ans = ''.join(res)
print(ans)