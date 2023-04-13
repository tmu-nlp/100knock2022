f = open('popular-names.txt', 'r')
f_unique = open('knock17_output.txt', 'w')
data = f.readlines()
name_list = []
for line in data:
    line = line.strip().split('\t')
    my_name = line[0]
    name_list.append(my_name)

name_set = set(name_list)
name_set = sorted(name_set)
for name in name_set:
    f_unique.write(name + '\n')
f.close()
f_unique.close()