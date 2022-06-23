"""
> 64の実行結果を用い，意味的アナロジー（semantic analogy）と文法的アナロジー（syntactic analogy）の正解率を測定せよ

意味的アナロジー：64で得られたcapital-common-countriesセクションなどの 類推結果一致数/サンプル数
文法的アナロジー：64で得られたgram1-adjective-to-adverbセクションなどの 類推結果一致数/サンプル数
"""

sem_num = 0
sem_correct = 0
syn_num = 0
syn_correct = 0

with open("questions-words-add.txt", "r") as f:
    for line in f:
        elems = line.split()

        # セクション名がgramから始まるなら文法的アナロジー正解数を計算
        if elems[0][:4] == "gram":
            if elems[4] == elems[5]:
                syn_correct += 1
            syn_num += 1

        # 意味的アナロジー正解数を計算
        else:
            if elems[4] == elems[5]:
                sem_correct += 1
            sem_num += 1

print("意味的アナロジー正解率:", sem_correct / sem_num)
print("文法的アナロジー正解率:", syn_correct / syn_num)


"""
意味的アナロジー正解率: 0.7308602999210734
文法的アナロジー正解率: 0.7400468384074942
"""
