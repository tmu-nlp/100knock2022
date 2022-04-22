f = open('popular-names.txt', 'r')
f_write = open('knock11_output.txt', 'w')
data = f.readlines()
for i, line in enumerate(data):
    data[i] = line.replace('\t', ' ')
f_write.writelines(data)
