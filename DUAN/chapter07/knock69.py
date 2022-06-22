import bhtsne
import numpy as np
from matplotlib import pyplot as plt
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

embedded = bhtsne.tsne(np.array(countries_vec).astype(np.float64), dimensions=2, rand_seed=123)
plt.figure(figsize=(10, 10))
plt.scatter(np.array(embedded).T[0], np.array(embedded).T[1])
for (x, y), name in zip(embedded, countries):
    plt.annotate(name, (x, y))
plt.show()
