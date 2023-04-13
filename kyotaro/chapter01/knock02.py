s = "パトカー"
t = "タクシー"
out = ""
index = 0

for i in range(len(s) + len(t)):
    if i % 2 == 0:
        out += s[index]

    else:
        out += t[index]
        index += 1

print(out)