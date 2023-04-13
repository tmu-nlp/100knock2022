N = int(input('N = '))

with open('popular-names.txt') as rf:
    lines = rf.readlines()

part = int(len(lines) / N)

for i in range(N):
    with open('separated_data' + str(i+1) + '.txt', 'w') as wf:
        wf.write(''.join(lines[i*part:(i+1)*part]))