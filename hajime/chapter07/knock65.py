
length = 0
cnt = 0
with open("64.txt", "r") as t_file:
    for line in t_file:
        length += 1
        line_list = line.strip().split(" ")
        if line_list[3] == line_list[4]:
            cnt += 1

print(float(cnt)/length)

"""
0.7358780188293083
"""
