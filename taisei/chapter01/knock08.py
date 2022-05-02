#008
def cipher(str):
    str_new = ''
    for i in range (len(str)):
        if (str[i] >= 'a' and str[i] <= 'z'):
            str_new += chr(219 - ord(str[i]))
        else:
            str_new += str[i]
            
    return str_new

print(cipher('abcdeABCD55'))
print(cipher(cipher('abcdeABCD55')))