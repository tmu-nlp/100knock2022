text = 'stressed'
ans = ''

for i in range(len(text)):
    ans += text[-(i+1)]

print(ans)