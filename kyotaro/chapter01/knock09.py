import random

s = "I couldn't believe that I could actually understand what I was reading : the phenomenal power of the human mind ."
s_div = s.split()
ans = []

for i in range(len(s_div)):
    if len(s_div[i]) > 4:
        ans.append(s_div[i][0] + ''.join(random.sample(s_div[i][1:len(s_div[i]) - 1], len(s_div[i]) - 2)) + s_div[i][len(s_div[i]) - 1])
        
    else:
        ans.append(s_div[i])

ans = ' '.join(ans)
print(ans)