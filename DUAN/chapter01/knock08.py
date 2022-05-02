def cipher(x):
    a = ''
    for i in x:
        if i.islower() == True:
            a += chr(219 - ord(i))
        else:
            a += i
    return(a)
print(cipher('Test aNd test'))