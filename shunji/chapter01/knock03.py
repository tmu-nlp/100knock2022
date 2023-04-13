text = 'Now I need a drink, alcoholic of course, after the heavy lectures involving quantum mechanics.'

ignore = [',', '.']
for sym in ignore:
    text = text.replace(sym, '')

word_list = text.split()
num_list = []

for word in word_list:
    word.rstrip(',')
    num_list.append(len(word))

print(num_list)