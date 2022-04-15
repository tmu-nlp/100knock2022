#00
str = 'stressed'
str_new = ''
for i in range (len(str)):
    str_new += str[len(str) - i - 1]
print(str_new)