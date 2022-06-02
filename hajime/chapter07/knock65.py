
length1 = 0
cnt1 = 0
with open("64-semantic.txt", "r") as t_file:
    for line in t_file:
        length1 += 1
        line_list = line.strip().split(" ")
        if line_list[3] == line_list[4]:
            cnt1 += 1

print("semantic analogy")
print(float(cnt1)/length1)

length2 = 0
cnt2 = 0
with open("64-syntactic.txt", "r") as t_file:
    for line in t_file:
        length2 += 1
        line_list = line.strip().split(" ")
        if line_list[3] == line_list[4]:
            cnt2 += 1
print("syntactic analogy")
print(float(cnt2)/length2)

"""
semantic analogy
0.7308602999210734
syntactic analogy
0.7400468384074942
"""
