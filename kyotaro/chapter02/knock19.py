from collections import defaultdict

name_dict = defaultdict(lambda: 0)
cnt = 0

with open("popular-names.txt", "r") as my_file:
    for line in my_file:
        line = line.strip()
        cnt += 1
        word = line.split("\t")
        name = word[0]
        name_dict[word[0]] += 1

name_dict = sorted(name_dict.items(), key = lambda x: x[1], reverse = True)

with open("knock19.txt", "w") as out_file:
    for key, value in name_dict:
        out_file.write(key + "\t" + str(value) + "\n")