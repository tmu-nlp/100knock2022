from knock30 import df
import pandas as pd
import matplotlib.pyplot as plt
import japanize_matplotlib #文字化け解消

freq = df['surface'].value_counts()#横軸
plt.hist(freq, bins=50)
plt.title('単語の出現頻度')
plt.xlabel('出現頻度')
plt.ylabel('単語の異なり数(種類数)')
plt.savefig('output38.png')
plt.show()
