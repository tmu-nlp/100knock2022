# knock65
# 64の実行結果を用い，意味的アナロジー（semantic analogy）と
# 文法的アナロジー（syntactic analogy）の正解率を測定せよ．
import re

if __name__ == "__main__":
    with open("output/output64.txt") as f:
        #文法的アナロジー:カテゴリ名にgram含まれてるカテゴリ
        #意味的アナロジー:gramが含まれていないカテゴリ
        #ref:https://www.soh-devlog.tokyo/nlp100-7-65/
        
        #意味的，文法的それぞれで事例数と正解数をカウント
        sem_cnt = 0
        syn_cnt = 0
        sem_cor = 0
        syn_cor = 0
        #今どのカテゴリか判別するフラグ
        is_syn = True
        for line in f:
            line = line.split()
            if len(line) != 6:
                if re.search(r"gram", line[1]):
                    is_syn = True
                    continue
                else:
                    is_syn = False
                    continue
            else:
                #文法的アナロジーのカテゴリのとき
                if is_syn:
                    syn_cnt += 1
                    #4列目が正解，5列目が求めた単語
                    if line[3] == line[4]:
                        syn_cor += 1
                #意味的アナロジーのカテゴリのとき
                else:
                    sem_cnt += 1
                    if line[3] == line[4]:
                            sem_cor += 1
        print(f"意味的アナロジーの正解率:{float(sem_cor/sem_cnt)}")
        print(f"文法的アナロジーの正解率:{float(syn_cor/syn_cnt)}")
"""
意味的アナロジーの正解率:0.7308602999210734
文法的アナロジーの正解率:0.7400468384074942
"""