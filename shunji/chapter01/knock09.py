import random

text = 'I couldn’t believe that I could actually understand what I was reading : the phenomenal power of the human mind .'

ignore = [',', '.', ':']
for sym in ignore:
    text = text.replace(sym, '')

word_list = text.split()

for i in range(len(word_list)):
    if len(word_list[i]) > 4:
        word_list[i] = word_list[i][0] + ''.join(random.sample(word_list[i][1:-1], len(word_list[i])-2)) + word_list[i][-1]

print(word_list)
        

