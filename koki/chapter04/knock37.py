from knock30 import sentences
import pandas as pd
import matplotlib.pyplot as plt
import japanize_matplotlib #文字化け解消
from collections import defaultdict

cnt_dict = defaultdict(int) #単語の出現回数をカウントする、int()は0を返す、「lambda: 0」と同様

for sentence in sentences:
    if '猫' in [info['surface'] for info in sentence]:#猫が含まれる一文か判定
        for info in sentence:
            if (info['surface'] != '猫') and (info['pos'] != '記号'):
                #if info['pos'] == '名詞': #例えば猫と関連性のある名詞とかに設定してあげれば実用的かも
                    cnt_dict[info['surface']] += 1 #単語「猫」と記号以外の単語と単語数をカウント

#cnt_dict = sorted(cnt_dict.items(), reverse=True)#これじゃソートできなかった

#pandasでソート
df = pd.DataFrame(cnt_dict, index = ['frequency'])
df = df.T
df = df.sort_values(by='frequency', ascending=False)

#上位10位を表示
df.head(10).plot(kind = 'bar', figsize = (10,8))
plt.legend(loc="lower left",bbox_to_anchor=(0.8,1.0))
plt.title('猫と共起頻度の高い単語上位10語')
plt.xlabel('単語')
plt.ylabel('出現回数')
plt.savefig('output37.png')
plt.show()
