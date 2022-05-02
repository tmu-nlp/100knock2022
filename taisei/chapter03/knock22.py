import re
f_uk = open('jawiki-uk.txt', 'r')
f_out = open('knock22_output.txt', 'w')
data = f_uk.readlines()
for line in data:
    line = line.strip()
    #rs = re.findall(r'\[\[Category:(.*?)(?:\|.*)?\]\]$', line) #(?:99)は99を除いて抽出 #(.*?)は最小のもの→[[Category:の右側で初めて|.*が出るまでのそれらの間のもの
    rs2 = re.search(r'\[\[Category:(.*?)(\|.*)?\]\]$', line)
    if rs2:
        #print(rs)
        f_out.write(rs2.group(1) + '\n')
f_uk.close()
f_out.close()