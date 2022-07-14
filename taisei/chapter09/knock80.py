import pandas as pd
from collections import defaultdict

def make_ids(train_data):
    count_dict = defaultdict(lambda: 0)

    for line in train_data['TITLE']:
        words = line.strip().split()
        for word in words:
            count_dict[word] += 1
    count_dict = sorted(count_dict.items(), key=lambda x:x[1], reverse=True)

    word2id = defaultdict(int)
    for i, (word, cnt) in enumerate(count_dict):
        if cnt <= 1:
            break
        word2id[word] = i + 1
    return word2id


def get_ids(text, word2id):
    words = text.strip().split()
    ids = []
    for word in words:
        ids.append(word2id[word])
    return ids

if __name__ == "__main__":
    train_data = pd.read_csv('../chapter06/output/train.txt', sep='\t', names=('CATEGORY', 'TITLE'))
    valid_data = pd.read_csv('../chapter06/output/valid.txt', sep='\t', names=('CATEGORY', 'TITLE'))
    test_data = pd.read_csv('../chapter06/output/test.txt', sep='\t', names=('CATEGORY', 'TITLE'))
    word2id = make_ids(train_data)
    text = valid_data['TITLE'][0]
    print(text)
    print(get_ids(text, word2id))

"""
Fracking Rules in North Carolina Tied to Koch, Halliburton, and ALEC
[3822, 619, 3, 272, 3660, 2580, 1, 0, 0, 14, 0]
"""