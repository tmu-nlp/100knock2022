class Morph():
    def __init__(self, surface, base, pos, pos1):
        self.surface = surface
        self.base = base
        self.pos = pos
        self.pos1 = pos1


class Chunk():
    def __init__(self, morphs, dst):
        self.morphs = morphs
        self.dst = int(dst)
        self.srcs = []


class Sentence():
    def __init__(self, chunks):
        self.chunks = chunks
        for i, chunk in enumerate(self.chunks):
            if chunk.dst != -1:
                self.chunks[chunk.dst].srcs.append(i)


data = []
morphs = []
chunks = []
sentences = []

with open('ai.ja.txt.parsed', 'r') as f:
    for line in f:
        if line[0] == '*':
            if len(data) > 0:
                chunks.append(Chunk(morphs, data[2][:-1]))
                morphs = []
            data = line.split()
        elif line != 'EOS\n':
            m = line.replace('\t', ',').split(',')
            morphs.append(Morph(m[0], m[7], m[1], m[2]))
        else:
            if len(data) > 0:
                chunks.append(Chunk(morphs, data[2][:-1]))
                sentences.append(Sentence(chunks))
                chunks = []
                morphs = []
                data = []

if __name__ == '__main__':
    for ch in sentences[1].chunks:
        for m in ch.morphs:
            print(m.surface, end='')
        print(', dst=' + str(ch.dst) + ', srcs=' + str(ch.srcs))
