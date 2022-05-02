from collections import defaultdict
f = open('popular-names.txt', 'r')
f_sort1 = open('knock19_output.txt', 'w')
data = f.readlines()
name_dic = defaultdict(lambda: 0)
for line in data:
    line = line.strip().split()
    name_dic[line[0]] += 1

#print(type(name_dic))
name_dic = sorted(name_dic.items(), reverse=True, key=lambda x:x[1]) #name_dicのvalueで大きい順にソートする。返されたものはリスト型になってる...
#print(type(name_dic))
for k, v in name_dic:
    f_sort1.write(str(v) + ' ' + k + '\n')
