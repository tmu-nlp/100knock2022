data = []

with open("popular-names.txt", "r") as my_file:
    for line in my_file:
        line = line.strip()
        word = line.split("\t")
        data.append(word)

data = sorted(data, key = lambda x: int(x[2]), reverse = True)

with open("knock18.txt", "w") as out_file:
    for i in range(len(data)):
        data[i] = "\t".join(data[i])
        out_file.write(data[i])
        out_file.write("\n")