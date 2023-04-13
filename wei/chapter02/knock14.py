with open('popular-names.txt', 'r', encoding='utf-8') as f:
    data = f.readlines()
    for i in range(10):
         print(f'{i+1}: {data[i].strip()}')