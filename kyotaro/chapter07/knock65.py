"""
64の実行結果を用い, 意味的アナロジー（semantic analogy）と文法的アナロジー（syntactic analogy）の正解率を測定せよ.
"""

with open("64.txt", "r") as data:
    sem_cnt = 0
    sem_cor = 0
    syn_cnt = 0
    syn_cor = 0
    flag = 0
    for line in data:
        line = line.strip().split()

        if "gram" not in line[1] and flag == 0:
            if line[0] != ":":
                sem_cnt += 1
                if line[3] == line[4]:
                    sem_cor += 1
        else:
            flag = 1
        
        if flag == 1:
            if line[0] != ":":
                syn_cnt += 1
                if line[3] == line[4]:
                    syn_cor += 1
    
    print(f'意味的アナロジー正解率：{sem_cor/sem_cnt:.3f}')
    print(f'文法的アナロジー正解率：{syn_cor/syn_cnt:.3f}')        