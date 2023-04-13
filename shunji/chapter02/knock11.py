with open('popular-names.txt', 'r') as f:
    for line in f:
        print(line.rstrip().replace('\t', ' '))