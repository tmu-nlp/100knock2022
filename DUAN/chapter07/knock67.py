import numpy as np
from sklearn.cluster import KMeans
from gensim.models import KeyedVectors

m = KeyedVectors.load_word2vec_format('./100knock2022/DUAN/chapter07/GoogleNews-vectors-negative300.bin', binary=True)

countries = set()
with open('./100knock2022/DUAN/chapter07/output.txt') as f:
    for line in f:
        line = line.split()
        if line[0] in ['capital-common-countries', 'capital-world']:
            countries.add(line[2])
        elif line[0] in ['currency', 'gram6-nationality-adjective']:
            countries.add(line[1])

countries = list(countries)
countries_vec = [m[country] for country in countries]
kmeans = KMeans(n_clusters=5)
kmeans.fit(countries_vec)

for i in range(5):
    cluster = np.where(kmeans.labels_ == i)[0]
    print('cluster', i)
    print(', '.join([countries[k] for k in cluster]))