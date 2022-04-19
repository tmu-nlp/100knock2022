def chipher(x):
    ans = ""
    for i in range(len(x)):
        if x[i].islower():
            ans += str(219 - ord(x[i]))
            
        else:
            ans += x[i]
    return ans

print(chipher("AcnB"))