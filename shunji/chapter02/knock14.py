N = int(input('N = '))

with open('popular-names.txt', 'r') as f:
    for i in range(N):
        print(f.readline().rstrip())