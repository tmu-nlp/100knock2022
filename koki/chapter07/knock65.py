with open('./results/output64.txt', 'r', encoding='utf-8_sig') as f:
    sem_cnt = 0  # 意味的アナロジー(semantic analogy)分母
    sem_cor = 0  # 意味的アナロジー分子
    syn_cnt = 0  # 文法的アナロジー(syntactic analogy)分母
    syn_cor = 0  # 文法的アナロジー分子

    for line in f:
        line = line.split('|')  # 「カテゴリ | 評価データ | 単語, 類似度」を | 区切りで3分割
        category = line[0]
        data = line[1].split()  # 評価データを4分割
        word, cos = line[2].split()  # 単語と類似度に分割

        # 文字列(カテゴリ)がgramで始まるかをstartswithで検査
        if not category.startswith('gram'):
            sem_cnt += 1
            if data[3] == word:  # knock64の予測が正解しているかどうか
                sem_cor += 1
        else:
            syn_cnt += 1
            if data[3] == word:
                syn_cor += 1

sem_accuracy = sem_cor / sem_cnt  # 正解率を計算(意味的アナロジー)
syn_accuracy = syn_cor / syn_cnt  # 正解率を計算(文法的アナロジー)

print('Accuracy(semantic analogy) : ', sem_accuracy)
print('Accuracy(syntactic analogy) : ', syn_accuracy)

'''
Accuracy(semantic analogy) :  0.7308602999210734
Accuracy(syntactic analogy) :  0.7400468384074942
'''
