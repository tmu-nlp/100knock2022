s = "Hi He Lied Because Boron Could Not Oxidize Fluorine. New Nations Might Also Sign Peace Security Clause. Arthur King Can."
ans = {}

s_div = s.split()

for i in range(len(s_div)):
    if i == 0 or 4 <= i <= 8 or 14 <= i <= 15 or i == 18:
        ans[s_div[i][0]] = i
        
    else:
        ans[s_div[i][:2]] = i

print(ans)