with open('popular-names.txt', 'r', encoding='utf-8') as f_in:
    name_list = []
    for line in f_in:
        line = line.strip().strip()
        name = line[0]
        name_list.append(name)
    names = set(name_list)
    print(f'total number of all names: {len(names)}')
    # total number of all names: 21


