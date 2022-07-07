with open('popular-names.txt', 'r', encoding='utf-8') as f_in, \
    open('knock12_col1.txt', 'w', encoding='utf-8') as f_col1, \
    open('knock12_col2.txt', 'w', encoding='utf-8') as f_col2:
    for line in f_in:
        line = line.split('\t')
        f_col1.write(line[0] + '\n')
        f_col2.write(line[1] + '\n')

