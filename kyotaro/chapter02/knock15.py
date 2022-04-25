import sys

N = int(sys.argv[1])
index = 0

my_file = open("popular-names.txt", "r").readlines()

for line in my_file[len(my_file) - N::]:
    line = line.strip()
    print(line)
    index += 1
    if index == N:
        break
