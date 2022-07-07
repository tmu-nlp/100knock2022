import pandas as pd
from collections import defaultdict
import string


class ID():
    def __init__(self, data):
        self.train_dict = defaultdict(int)
        self.id_list = []
        self.id_dict = dict()
        self.make_id(data)

    def make_id(self, data):
        for line in data:
            words = line.translate(table).split()
            for word in words:
                # if word != "":
                self.train_dict[word] += 1
        calc_dict = dict(self.train_dict)
        sort_list = sorted(calc_dict.items(), key=lambda x: x[1], reverse=True)
        for i, (trg_word, freq) in enumerate(sort_list):
            if freq >= 2:
                self.id_list.append((trg_word, i+1))
            else:
                self.id_list.append((trg_word, 0))
        self.id_dict = dict(self.id_list)

    def return_id(self, line):
        one_hot_vec = []
        words = line.strip().split(" ")
        for word in words:
            if word in self.id_dict.keys():
                one_hot_vec.append(self.id_dict[word])
            else:
                one_hot_vec.append(0)
        return one_hot_vec


table = str.maketrans(string.punctuation, ' '*len(string.punctuation))

header_name = ['TITLE', 'CATEGORY']
train_file = "../chapter06/train.txt"
train_data = pd.read_csv(train_file, header=None,
                         sep='\t', names=header_name)
w2id = ID(train_data['TITLE'])
test_vec = w2id.return_id(train_data["TITLE"][10])

if __name__ == "__main__":
    # print(test_vec)
    print(w2id.id_dict)
