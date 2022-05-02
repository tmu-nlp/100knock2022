words = set()

with open('popular-names.txt', 'r') as f:
    for line in f:
        words.add(line.split('\t')[0])

print(words)


