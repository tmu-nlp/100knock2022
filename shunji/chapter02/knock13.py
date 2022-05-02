with open('col1.txt', 'r') as rf1:
    col1 = rf1.readlines()
with open('col2.txt', 'r') as rf2:
    col2 = rf2.readlines()
with open('2cols.txt', 'w') as wf:
    for i in range(len(col1)):
        wf.write(col1[i].rstrip() + '\t' + col2[i])