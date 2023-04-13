from collections import defaultdict

name_dict = defaultdict(lambda: 0)

with open("popular-names.txt", "r") as my_file:
    for line in my_file:
        line = line.strip()
        word = line.split("\t")
        name = word[0]
        name_dict[word[0]] += 1

print(len(name_dict))