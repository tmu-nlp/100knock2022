'''
67. k-means clustering
国名に関する単語ベクトルを抽出し、k-meansクラスタリングをクラスタ数k=５として実行
https://scikit-learn.org/stable/auto_examples/cluster/plot_kmeans_assumptions.html#sphx-glr-auto-examples-cluster-plot-kmeans-assumptions-py
'''
import pickle
import pandas as pd
import numpy as np
from sklearn.cluster import KMeans
from matplotlib import pyplot as plt
from knock60 import *


def get_country_name(line_list, category):
    countries = set()
    for line in line_list:
        if line.startswith(f': {category}'):
            ctg = category
        elif line.startswith(':'):
            ctg = 'others'
        else:
            if ctg == category:
                country_1 = line.split()[1]
                country_3 = line.split()[3]
                countries.add(country_1)
                countries.add(country_3)

            elif ctg == 'others':
                continue
    countries = list(countries)

    return countries



if __name__ == '__main__':
    import datetime
    start = datetime.datetime.now()
    print(f'start running time: {start}.')

    wv = load_wv()    # load KeyedVectors
    with open('../data/questions-words.txt','r', encoding='utf-8') as f1:
        lines = f1.readlines()
    countries_1 = get_country_name(lines, 'capital-common-countries')
    countries_2 = get_country_name(lines, 'capital-world')
    countries_3 = get_country_name(lines, 'currency')
    final_countries = list(set(countries_1 + countries_2 + countries_3))

    # 国名のベクトルを取得
    vec_ctrs = [wv[country] for country in final_countries]
    vec_ctrs_arr = np.array(vec_ctrs)
    # print(vec_ctrs_arr.shape)   (144, 300)

    # k-means clustering
    # fit(X,y=None):
    # y:ignored, present for API consistency, compute k-means clustering, return fitted estimator object.
    # fit_predict(X,y):
    # compute cluster centers and predict cluster index for each sample.
    # return ndarray of shape (n_samples,):index of cluster each sample belongs to.

    y_pred = KMeans(n_clusters=5, random_state=112).fit(vec_ctrs_arr)
    for i in range(5):
         cluster = np.where(y_pred.labels_== i)[0]
         print('cluster', i)
         print(','.join([final_countries[k] for k in cluster]))

    # all rows in 1st column, all rows in 2nd column, not ok if ndarray(n>=2)
    # plt.scatter(vec_ctrs_arr[:,0], vec_ctrs_arr[:,1], c=y_pred.labels_)
    #
    # plt.show()
    # plt.savefig('./knock67_fit.jpg')
    #
    #
    # np.save('./vec_ctrs_arr.npy', vec_ctrs_arr)
    # with open('./total_num_ctrs.txt', 'wb') as f2:
    #     pickle.dump(final_countries, f2)   #持久化包括python自定义在内的各种数据，将python对象保存



    end = datetime.datetime.now()
    print(f'end running time: {end}.')
    print(f'running time : {end-start} seconds.')




