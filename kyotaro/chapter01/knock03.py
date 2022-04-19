s = "Now I need a drink, alcoholic of course, after the heavy lectures involving quantum mechanics."
ans = []

s_div = s.split()

for i in range(len(s_div)):
    ans.append(len(s_div[i]))

print(ans)