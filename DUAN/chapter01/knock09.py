import random

def Typoglycemia(sentence):
    a = sentence.split(' ')
    b = list()
    for i in a:
        if len(i) <= 4:
            b.append(i)
        else:
            temp = i[1:len(x)-1]
            temp = ''.join(random.sample(temp,len(temp)))
            b.append(i[0] + temp + i[len(i)-1])
    return ' '.join(map(str,b))

x = r"I couldn't believe that I could actually understand what I was reading : the phenomenal power of the human mind ."
print(Typoglycemia(x))