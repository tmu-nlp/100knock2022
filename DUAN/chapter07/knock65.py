f = open('./100knock2022/DUAN/chapter07/output.txt')
sem_cnt = 0; sem_cor = 0
syn_cnt = 0; syn_cor = 0
for line in f:
    line = line.split()
    if not line[0].startswith('gram'):
        sem_cnt += 1
        if line[4] == line[5]:
            sem_cor += 1
    else:
        syn_cnt += 1
        if line[4] == line[5]:
            syn_cor += 1
print(f'semantic analogy: {sem_cor/sem_cnt:.3f}')
print(f'syntactic analogy: {syn_cor/syn_cnt:.3f}') 