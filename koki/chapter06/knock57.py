import numpy as np

features = x_train.columns.values  # 学習データの特徴量

class_name, coef = model.classes_[0], model.coef_[0]
index = [i for i in range(1, 11)]

top_10 = features[np.argsort(-coef)[:10]]  # 降順
worst_10 = features[np.argsort(coef)[:10]]

df_top_10 = pd.DataFrame(top_10, columns=[f'重みの高い特徴量トップ10（クラス名: {class_name}）'], index = index)
df_worst_10 = pd.DataFrame(worst_10, columns=[f'重みの低い特徴量トップ10（クラス名: {class_name}）'], index= index)

'''
for class_name, coef in zip(model.classes_, model.coef_):
    index = [i for i in range(1, 11)]
    top_10 = features[np.argsort(-coef)[:10]]
    worst_10 = features[np.argsort(coef)[:10]]
    df_top_10 = pd.DataFrame(top_10, columns=[f'重みの高い特徴量トップ10（クラス名: {class_name}）'], index = index)
    df_worst_10 = pd.DataFrame(worst_10, columns=[f'重みの低い特徴量トップ10（クラス名: {class_name}）'], index= index)
'''

df_top_10.to_csv('./results/output_top10.csv')
df_worst_10.to_csv('./results/output_worst10.csv')

df_top_10     
