import re
f = open('jawiki-uk.txt', 'r')
f_out = open('knock24_output.txt', 'w')
data = f.readlines()
for line in data:
    line = line.strip()
    rs = re.findall(r'\[\[ファイル:(.*?)\]\]$', line)
    #rs2 = re.search(r'\[\[ファイル:(.*?)\]\]$', line)
    if rs:
        refer = rs[0].split('|')[0]
        f_out.write(f'{refer}\n')
        #print(rs2.group(1).split('|')[0])
f.close()
f_out.close()
