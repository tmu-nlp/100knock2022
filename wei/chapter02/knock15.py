with open('popular-names.txt', 'r', encoding='utf-8') as f:
    data = f.readlines()
    for i, line in enumerate(data[::-1][:10]):
         print(f'{len(data)-i}: {line.strip()}')

