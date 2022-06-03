from graphviz import Digraph
dg=Digraph(format='png')

class Morph:
    def __init__(self,surface,base,pos,pos1):
        self.surface=surface
        self.base=base
        self.pos=pos
        self.pos1=pos1

    def __str__(self):
        x="surface:{0}\tbase:{1}\tpos:{2}\tpos1:{3}"
        return x.format(self.surface,self.base,self.pos,self.pos1)

class Chunk:
    def __init__(self):
        self.counter=0
        self.morphs=[]
        self.dst=-1
        self.srcs=[]

morph=[]
prin=0
count=0
chunks=dict()
chunks[count]=Chunk()
with open('./100knock2022/DUAN/chapter05/ai.ja.txt.parsed',encoding='utf-8')as f:
    for line in f:
        surface=line.split('\t')
        chunks[count+1]=Chunk()
        if len(surface)>1:
            other=surface[1].split(',')
            Base=other[6]
            Pos=other[0]
            Pos1=other[1]
            morph.append(Morph(surface[0],Base,Pos,Pos1))
            if Pos!='記号':
                chunks[count].morphs.append(surface[0])#chunk.morphs
        elif 'EOS' not in line:
            kakari=line.split(' ')
            count=int(kakari[1])
            dst=kakari[2].replace('D','')
            chunks[count].dst=int(dst)                    #chunk.dst
            chunks[count].counter=count
        else:
            for number in range(count+1):
                if chunks[number].dst != -1:
                    chunks[chunks[number].dst].srcs.append(number)
                dg.node(''.join(chunks[number].morphs))

            for result in range(count+1):
                x="{0}{1}\t{2}{3}"
                if chunks[result].dst == -1:
                    saki=''
                else:
                    saki=''.join(chunks[chunks[result].dst].morphs)
                if saki:
                    moto=''.join(chunks[result].morphs)
                    dg.edge(moto,saki)
            if dg:
                prin+=1
                filename='nock44'+str(prin)
                dg.render(filename)
                dg.clear()

            for ketu in range(count+1):
                chunks[ketu].morphs=[]
                chunks[ketu].srcs=[]
            count=0

'''
print(pairs)
g = pydot.graph_from_edges(pairs)
g.write_png('./100knock2022/DUAN/chapter05/knock44.png', prog='dot')
'''