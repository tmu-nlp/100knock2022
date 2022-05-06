f_col1 = open('col1.txt', 'r')
f_col2 = open('col2.txt', 'r')
f_join = open('knock13_output.txt', 'w')
data_col1 = f_col1.readlines()
data_col2 = f_col2.readlines()
for i in range (len(data_col1)):
    join_word = data_col1[i].strip() + '\t' + data_col2[i].strip() + '\n'
    f_join.write(join_word)
f_col1.close()
f_col2.close()
f_join.close()