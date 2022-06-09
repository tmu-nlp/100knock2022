from gensim.models import KeyedVectors

if __name__ == "__main__":
    with open("./output/knock64_output.txt", "r") as f_data:
        data = f_data.readlines()

    sem_cor = 0  #意味的アナロジーの正解数
    sem_dif = 0  #意味的アナロジーの誤り数
    syn_cor = 0  #文法的アナロジーの正解数
    syn_dif = 0  #文法的アナロジーの誤り数
    is_sem = True  #意味的のほうか文法的のほうか


    for line in data:
        words = line.strip().split()
        if words[0] == ":":
            if words[1] == "gram1-adjective-to-adverb": #ここで意味的と文法的が切り替わる(8875行目)
                is_sem = False
            continue
        
        if is_sem:
            if words[3] == words[4]:
                sem_cor += 1
            else:
                sem_dif += 1
        else:
            if words[3] == words[4]:
                syn_cor += 1
            else:
                syn_dif += 1
    print(f'意味的アナロジーの正解率  {sem_cor / (sem_cor + sem_dif)}')
    print(f'文法的アナロジーの正解率  {syn_cor / (syn_cor + syn_dif)}')

"""
意味的アナロジーの正解率  0.7308602999210734
文法的アナロジーの正解率  0.7400468384074942
"""
    



        

