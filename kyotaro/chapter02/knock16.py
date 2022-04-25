import sys

N = int(sys.argv[1])
index = 0
file_num = 0

my_file = open("popular-names.txt", "r").readlines()

for line in my_file:
    with open("sp{filenum}.txt".format(filenum = file_num), "a+") as sss:
        sss.write(line)
        index += 1
        if index == N:
            index = 0
            file_num += 1
            continue