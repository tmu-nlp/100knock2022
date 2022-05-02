text = 'Hi He Lied Because Boron Could Not Oxidize Fluorine. New Nations Might Also Sign Peace Security Clause. Arthur King Can.'

ignore = [',', '.']
for sym in ignore:
    text = text.replace(sym, '')

word_list = text.split()

selected_num = [1, 5, 6, 7, 8, 9, 15, 16, 19]
order = {}

for i in range(len(word_list)):
    if i+1 in selected_num:
        order[word_list[i][0]] = i
    else:
        order[word_list[i][:2]] = i

print(order)
