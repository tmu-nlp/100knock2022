f = open('popular-names.txt', 'r')
f_col1 = open('col1.txt', 'w')
f_col2 = open('col2.txt', 'w')
data = f.readlines()
for i, line in enumerate(data):
    line = line.strip().split('\t')
    f_col1.write(line[0] + '\n')
    f_col2.write(line[1] + '\n')
