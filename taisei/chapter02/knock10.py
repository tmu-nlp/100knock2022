f = open('popular-names.txt', 'r')
f_write = open('knock10_output.txt', 'w')
data = f.readlines()
f_write.write(str(len(data)))
