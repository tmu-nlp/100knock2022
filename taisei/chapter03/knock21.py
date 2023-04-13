import re
f_uk = open('jawiki-uk.txt', 'r')
f_out = open('knock21_output.txt', 'w')
data = f_uk.readlines()
for line in data:
    rs = re.search(r'\[\[Category:.*\]\]', line)
    if rs:
        f_out.write(rs.group() + '\n')
f_uk.close()
f_out.close()