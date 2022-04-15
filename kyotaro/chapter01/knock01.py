s = "パタトクカシーー"
out = ""

for i in range(len(s) - 1):
    if i % 2 == 0:
        out += s[i]

print(out)