with open('popular-names.txt', 'r', encoding='utf-8') as f_in, \
    open('knock11_out.txt', 'w', encoding='utf-8') as f_out:
    for line in f_in:
        line = line.replace('\t', ' ')
        f_out.write(line)
