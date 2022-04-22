import sys
f = open('popular-names.txt', 'r')
f_head = open('knock14_output.txt', 'w')
data = f.readlines()
n = int(sys.argv[1])
for i in range (n):
    f_head.write(data[i])
