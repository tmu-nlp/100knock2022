'''
65. アナロジータスクでの正解率
64の実行結果を用い、意味的アナロジー(semantic analogy)と文法的アナロジー(syntactic analogy)の正解率を測定
意味的アナロジー: 前から5カテゴリで、例えばAtnens Greece Beijing China　China	0.7235778570175171
文法的アナロジー: 第6カテゴリから最後まで、例えばplay playing slow slowing
'''

import numpy as np

if __name__ == '__main__':
    with open('./knock64.txt', 'r') as f:
        res = f.readlines()

    syntactic, semantic = [], []
    for line in res:
        if line.startswith(': gram'):
            ctg = 'syntactic'
        elif line.startswith(':'):
            ctg = 'semantic'
        else:
            word = line.split()[3]
            word_pred = line.split()[4]
            if ctg == 'syntactic':
                syntactic.append((word == word_pred))
            elif ctg == 'semantic':
                semantic.append((word == word_pred))
            else:
                print('No ctg')


    syntactic_acc = np.array(syntactic).mean()
    semantic_acc = np.array(semantic).mean()
    print(f'semantic_acc_rate:{semantic_acc:.4f}')
    print(f'syntactic_acc:{syntactic_acc:.4f}')
# semantic_acc_rate:0.7309
# syntactic_acc:0.7400




