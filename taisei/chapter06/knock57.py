import pickle

with open("./output/knock52_lr_model", "rb") as f_model:
    lr = pickle.load(f_model) #ロジスティック回帰の学習済みモデル
with open("./output/knock51_lr_voc", "rb") as f_voc:
    voc = pickle.load(f_voc) #vocは(単語, その単語の重みでのindex)の形が単語数分ある
catego_list = ['b', 'e', 'm', 't']

#lr.coef_は学習した重み。カテゴリが4つだから4行*単語数
for i, catego in enumerate(catego_list):
    now_coefs = lr.coef_[i] #カテゴリごとの重み
    feature_dic = dict()
    for word, index in voc.items():
        feature_dic[word] = now_coefs[index]

    print(f'---------{catego}---------')
    print(f'Top 10 of {catego}')
    for word, weight in sorted(feature_dic.items(), key=lambda x:x[1], reverse=True)[:10]:
        print(f"{word}    {weight}")
    print()
    
    print(f'Worst 10 of {catego}')
    for word, weight in sorted(feature_dic.items(), key=lambda x:x[1], reverse=False)[:10]:
        print(f"{word}    {weight}")
    print()