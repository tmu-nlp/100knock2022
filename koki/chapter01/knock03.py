text = 'Now I need a drink, alcoholic of course, after the heavy lectures involving quantum mechanics.'
list_word = []
list_n_word = []
word = ''

list_word = text.split(' ')

for word in list_word:
    list_n_word.append(len(word))

print(list_n_word)
