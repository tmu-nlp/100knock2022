import re
import pydot
from IPython.display import Image

class Morph(object):
    srcs = -1
    def __init__(self, surface, base, pos, pos1):
        self.surface = surface
        self.base = base
        self.pos = pos
        self.pos1 = pos1

class Chunk(object):
    def __init__(self, morphs, dst, srcs):
        self.morphs = morphs
        self.dst = dst
        self.srcs = srcs

    def get_surface(self):
        surface = [morph.surface for morph in self.morphs]
        surface = ('').join(surface)
        return re.sub(r'[、。]', '', surface)

    def get_pos(self, pos):
        return [morph for morph in self.morphs
                if morph.pos == pos]

    def apply_index(self, index):
        for morph in self.morphs:
            morph.chunk_id = index

    def has_pos(self, pos):
        return pos in [morph.pos for morph in self.morphs]

    def replace_pos(self, pos, X):
        surface = ''
        for morph in self.morphs:
            if morph.pos == pos:
                surface += X
            else:
                surface += morph.surface
        return re.sub(r'[、。]', '', surface)

def get_chunks(text):
    data = []
    for sentence in re.findall(r'(\n[\s\S]*?EOS)', text):
        chunks = []
        for clause in re.findall(
                r'\* (\d*) (-?\d+).*?\n([\s\S]*?)(?=\n\*|\nEOS)', sentence):  
            morphs = []
            srcs = re.findall(
                r'\* (\d*) ' + clause[0] + r'D.*?\n[\s\S]*?(?=\n\*|\nEOS)',
                sentence)
            for line in re.findall(r'(.*?)\t(.*?)(?:$|\n)', clause[2]): 
                surface = line[0]
                feature = line[1].split(',')
                morph = Morph(surface, feature[6], feature[0], feature[1])
                morphs.append(morph)
            chunk = Chunk(morphs, int(clause[1]), list(map(int, srcs)))
            chunks.append(chunk)
        data.append(chunks)
    return data

graph = pydot.Dot(graph_type='digraph')
chunks = ''

with open('./100knock2022/DUAN/chapter05/ai.ja.txt.parsed', 'r') as f:
    text = f.read()
    chunks = get_chunks(text)[7]

for chunk in chunks:
    if chunk.dst == -1:
        continue
    surface = chunk.get_surface()
    dst_surface = chunks[chunk.dst].get_surface()
    edge = pydot.Edge(surface, dst_surface)
    graph.add_edge(edge)

graph.write('./100knock2022/DUAN/chapter05/knock44.png', format="png")
Image(graph.create(format='png'))
