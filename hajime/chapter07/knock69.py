import bhtsne
from knock67 import *
import matplotlib.pyplot as plt
import numpy as np

emb = bhtsne.tsne(np.array(country_vec).astype(
    np.float64), dimensions=2, rand_seed=20010101)
plt.figure(figsize=(10, 10))
plt.scatter(np.array(emb[:, 0]), np.array(emb[:, 1]))
for (x, y), name in zip(emb, country_list_in):
    plt.annotate(name, (x, y))
plt.savefig("69.png")
