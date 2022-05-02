import sys
f = open('popular-names.txt', 'r')
f_tail = open('knock15_output.txt', 'w')
data = f.readlines()
n = int(sys.argv[1])
for i in range(n)[::-1]: #スライスによりiは 0 ~ n-1 じゃないで n-1 ~ 0 になる
    f_tail.write(data[len(data) - i - 1])
