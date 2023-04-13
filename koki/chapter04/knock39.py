from cmath import log
from tkinter.tix import Tree
from knock30 import df
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import japanize_matplotlib #文字化け解消

freq = df['surface'].value_counts()
frequency = np.array(freq)#y軸　単語の出現頻度回数
rank = [r+1 for r in range(len(freq))]#x軸　出現頻度ランキング 要素毎のlen+1が順位

plt.figure(figsize=(10,8))
plt.title('zipfの法則')
plt.xlabel('単語の出現頻度順位')
plt.ylabel('単語の出現頻度')
#plt.plot(rank, frequency) #折れ線グラフ
plt.scatter(rank, frequency) #散布図
plt.xscale('log')
plt.yscale('log')
plt.savefig('output39.png')
plt.show()
