s = 'パタトクカシーー'
new_s = ''
i = 1
for char in s:
    if i < len(s):
        new_s += s[i]
        i += 2
print(new_s)