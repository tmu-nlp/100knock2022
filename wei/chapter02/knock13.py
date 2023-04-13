with open('knock13_out.txt', 'w', encoding='utf-8') as f_out, \
    open('knock12_col1.txt', 'r', encoding='utf-8') as f_col1, \
    open('knock12_col2.txt', 'r', encoding='utf-8') as f_col2:
    data_col1  =f_col1.readlines()
    data_col2 = f_col2.readlines()


    for i in range(len(data_col1)):
            f_out.write(f'{data_col1[i].strip()}\t{data_col2[i].strip()}\n')
