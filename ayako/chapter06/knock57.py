# knock57
# 52で学習したロジスティック回帰モデルの中で，
# 重みの高い特徴量トップ10と，重みの低い特徴量トップ10を確認せよ．
import pickle

if __name__ == "__main__":
    lr = pickle.load(open("model.pkl", "rb"))
    vocabulary = pickle.load(open("vocabulary.pkl", "rb"))#[単語]=単語id(重みの出力順に対応)になる辞書

    #coef_で各カテゴリの重みを取得
    weights_dic = lr.coef_

    categories = ["business", "entertainment", "health", "science and technology"]
    
    for i, weights in enumerate(weights_dic):#各カテゴリごとに
        #key:単語，val:重みの辞書を作成
        features = dict()
        for word, index in vocabulary.items():
            features[word] = weights[index]#weightsはそのカテゴリにおける各単語の重みリスト
        
        print(f"-------category:{categories[i]}--------")
        print("[Best10]")
        for word, weight in sorted(features.items(), key=lambda x:x[1], reverse=True)[:10]:#重みの値の降順に
            print(f"{word}\t{weight}")
        
        print("[Worst10]")
        for word, weight in sorted(features.items(), key=lambda x:x[1])[:10]:#重みの値の昇順に
            print(f"{word}\t{weight}")

        print("\n")