import re
f = open('jawiki-uk.txt', 'r')
f_out = open('knock26_output.txt', 'w')
data = f.readlines()
for line in data:
    line = line.strip()
    rs = re.match(r'\|(.*?)(\s\=\s*)(.*?)$', line)
    if rs:
        field_name = rs.group(3)
        rs_empha_delete = re.sub(r'\'+', '', field_name)
        f_out.write(f'{rs.group(1)} : {rs_empha_delete}\n')

f.close()
f_out.close()