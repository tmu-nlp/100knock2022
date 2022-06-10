'''
68. Ward法によるクラスタリング
国名に関する単語ベクトルに対し，Ward法による階層型クラスタリングを実行せよ．
さらに，クラスタリング結果をデンドログラムとして可視化せよ．
blog_cn:
https://blog.csdn.net/weixin_42887138/article/details/117708688?spm=1001.2101.3001.6650.1&utm_medium=distribute.pc_relevant.none-task-blog-2%7Edefault%7ECTRLIST%7ERate-1-117708688-blog-7801764.pc_relevant_antiscanv3&depth_1-utm_source=distribute.pc_relevant.none-task-blog-2%7Edefault%7ECTRLIST%7ERate-1-117708688-blog-7801764.pc_relevant_antiscanv3&utm_relevant_index=1
doc:
https://docs.scipy.org/doc/scipy/reference/cluster.hierarchy.html#
'''
import numpy as np
import pickle
from scipy.cluster.hierarchy import dendrogram, linkage, fcluster
from matplotlib import pyplot as plt



# load vectors,countries from npy,pkl
vec_ctrs = np.load('./vec_ctrs_arr.npy')  # (144, 300)
ctr_list = pickle.load(open('./total_num_ctrs.txt','rb'))

# 階層型クラスタリングの実施
Z = linkage(vec_ctrs, method='ward')
# arguments: (n samples,m dim), ward->离差平方和距离
# return: shape->(n-1,4): 每列元素代表的意思：
# 前两个元素是每一步合并的cluster的index，第三个元素是被合并cluster之间的距离，第三个元素表示合并后的cluster中元素的个数
#print('Z:\n', Z)
f = fcluster(Z, t=28.8, criterion='distance')    # t=144/5
# form flat clusters from hierarchical clustering defined by the given linkage matrix
# return: ndarrary: an arrary of length n. T[i]是原始观测值i所属的平面簇number。
#print('f:\n', f)

fig = plt.figure(figsize=(12,6))
dendrogram(Z, labels=ctr_list)

plt.title('country_dendrogram')
plt.savefig('./knock68.png')
plt.show()