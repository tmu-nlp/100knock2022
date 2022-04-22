f = open('popular-names.txt', 'r')
f_sort = open('knock18_output.txt', 'w')
data = f.readlines()
line_list = []

for i, line in enumerate(data):
    line = line.strip().split('\t')
    line[2] = int(line[2]) #strのままだとsortの時に変になるからintに変換
    line_list.append(line)

sort_list = sorted(line_list, reverse=True, key=lambda x: x[2]) #line_listは 4×2780 の配列。各行の3要素目([2])に関してsortする https://qiita.com/happyisland44/items/2f078c3164ccfcac561f
for i in range(len(sort_list)):
    sort_list[i][2] = str(sort_list[i][2])

for line in sort_list:
    f_sort.write("\t".join(line) + '\n') #タブ区切りで連結