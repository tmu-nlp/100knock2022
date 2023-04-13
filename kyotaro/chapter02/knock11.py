with open("popular-names.txt") as f:
    for line in f:
        print(line.strip().replace("\t", " "))