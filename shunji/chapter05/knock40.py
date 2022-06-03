class Morph():
    def __init__(self, surface, base, pos, pos1):
        self.surface = surface
        self.base = base
        self.pos = pos
        self.pos1 = pos1


morph_list = []
sentences = []

with open('ai.ja.txt.parsed', 'r') as f:
    for line in f:
        if line[0] == '*':
            continue
        elif line != 'EOS\n':
            data = line.replace('\t', ',').split(',')
            morph_list.append(Morph(data[0], data[7], data[1], data[2]))
        else:
            sentences.append(morph_list)
            morph_list = []

for m in sentences[2]:
    print(vars(m))  # vars(obj) オブジェクトの__dict__を呼び出す
