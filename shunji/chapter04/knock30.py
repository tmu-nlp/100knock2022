morphs = []
sentences = []

with open('neko.txt.mecab', 'r') as f:
    for line in f:
        if line != 'EOS\n':
            if line == '\n':
                continue
            data = line.replace('\t', ',').split(',')
            m = {'surface': data[0], 'base': data[7], 'pos': data[1], 'pos1': data[2]}
            morphs.append(m)
        else:
            sentences.append(morphs)
            morphs = []

if __name__ == '__main__':
    print(*sentences, sep='\n')


