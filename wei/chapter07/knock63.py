'''
63. 加法構成性によるアナロジー
new_v = Spain - Madrid + Athens
output top 10 (vector, sim)
'''

from knock60 import *

if __name__ == '__main__':
    wv = load_wv()
    result = wv.most_similar(positive=['Spain', 'Athens'], negative=['Madrid'])
    for i,v in enumerate(result):
        print(f'top {i+1} is {v}.')

        '''
top 1 is ('Greece', 0.6898480653762817).
top 2 is ('Aristeidis_Grigoriadis', 0.560684859752655).
top 3 is ('Ioannis_Drymonakos', 0.5552908778190613).
top 4 is ('Greeks', 0.545068621635437).
top 5 is ('Ioannis_Christou', 0.5400862097740173).
top 6 is ('Hrysopiyi_Devetzi', 0.5248445272445679).
top 7 is ('Heraklio', 0.5207759737968445).
top 8 is ('Athens_Greece', 0.516880989074707).
top 9 is ('Lithuania', 0.5166865587234497).
top 10 is ('Iraklion', 0.5146791338920593).'''
