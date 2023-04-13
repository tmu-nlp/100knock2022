N = int(input('N = '))

with open('popular-names.txt', 'r') as f:
    lines = f.readlines()
    for line in lines[len(lines)-N:]:
        print(line.rstrip())