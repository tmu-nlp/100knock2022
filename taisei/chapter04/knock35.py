import knock30
from collections import defaultdict
count_dict = defaultdict(lambda: 0)

for line in knock30.morpheme_line_list:
    for morpheme in line:
        if morpheme["品詞"] == "記号":
            continue
        count_dict[morpheme["基本形"]] += 1

count_dict_sort = sorted(count_dict.items(), key=lambda x:x[1], reverse=True)

with open("knock35_output.txt", "w") as f_out:
    for my_key, my_value in count_dict_sort:
        f_out.write(f'{my_key} {my_value}\n')
