import random

def Typoglycemia(sen):
    a = sen.split(' ')
    b = list()
    for i in a:
        if len(i) <= 4:
            b.append(i)
        else:
            c = i[1:len(x)-1]
            c = ''.join(random.sample(c,len(c)))
            b.append(i[0] + c + i[len(i)-1])
    return ' '.join(map(str,b))

x = "I couldn't believe that I could actually understand what I was reading : the phenomenal power of the human mind ."
print(Typoglycemia(x))
