def n_gram(n, text):
    l = []
    if type(text) == str:
        for i in range(len(text) - n + 1):
            l.append(''.join(text[i:i+n]))
    else:
        for i in range(len(text) - n + 1):
            l.append(' '.join(text[i:i+n]))
    return l

text = 'I am an NLPer'
word_list = text.split()
fig_list = ''.join(word_list)

w_bi_gram = n_gram(2, word_list)
f_bi_gram = n_gram(2, fig_list)

print(w_bi_gram)
print(f_bi_gram)