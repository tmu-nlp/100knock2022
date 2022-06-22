'''
69. t-SNEによる可視化
ベクトル空間上の国名に関する単語ベクトルをt-SNEによる可視化
'''

import numpy as np
import pickle
from matplotlib import pyplot as plt
import bhtsne



# load vectors,countries from npy,pkl
vec_ctrs = np.load('./vec_ctrs_arr.npy')  # (144, 300)
ctr_list = pickle.load(open('./total_num_ctrs.txt','rb'))

embedded = bhtsne.tsne(np.array(vec_ctrs).astype(np.float64), dimensions=2, rand_seed=112)
plt.figure(figsize=(10,10))
plt.scatter(np.array(embedded).T[0], np.array(embedded).T[1])
for (x, y), name in zip(embedded, ctr_list):
    plt.annotate(name, (x,y))
plt.show()
plt.savefig('./knock69.png')