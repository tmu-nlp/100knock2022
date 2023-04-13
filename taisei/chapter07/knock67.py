import pickle
from sklearn.cluster import KMeans
# k-meansの参考サイト
# https://pythondatascience.plavox.info/scikit-learn/%E3%82%AF%E3%83%A9%E3%82%B9%E3%82%BF%E5%88%86%E6%9E%90-k-means

with open("./output/knock60_word2vec", "rb") as f:
    model = pickle.load(f)

with open("questions-words.txt", "r") as f_data:
    data = f_data.readlines()

country_set = set()
is_capital = False  # : capital-hogeの行は[1]と[3]が国名っぽいから、capital下のデータかの判断に使う
is_curr = False  # : currencyと: gram6-...の行は[2]が国名っぽいから、上と同様

for line in data:
    line = line.strip().split()
    if line[0] == ":":
        if line[1] in ['capital-common-countries', 'capital-world']:
            is_capital = True
            is_curr = False
        elif line[1] in ["currency", "gram6-nationality-adjective"]:
            is_capital = False
            is_curr = True
        else:
            is_capital = False
            is_curr = False

    else:
        if is_capital:
            country_set.add(line[1])
            country_set.add(line[3])
        elif is_curr:
            country_set.add(line[2])

in_countrys = []  # 国名
in_countrys_vec = []  # 単語ベクトル
for country in country_set:
    vec = model[country]
    in_countrys.append(country)
    in_countrys_vec.append(vec)


if __name__ == "__main__":
    same_cluster = dict()
    for i in range(5):  # クラスタiの国名をまとめるための辞書
        same_cluster[i] = []

    kmeans = KMeans(n_clusters=5, random_state=0)
    y_kmeans = kmeans.fit_predict(in_countrys_vec)
    for country, cluster_num in zip(in_countrys, y_kmeans):
        same_cluster[cluster_num].append(country)
    print(y_kmeans)

    for i in range(5):
        print(f'----------cluster {i}----------')
        print(", ".join(same_cluster[i]))

"""
----------cluster 0----------
Pakistan, Iraq, Bahrain, Morocco, Libya, Egypt, Syria, China, Indonesia, Cambodia, Thailand, Iran, Lebanon, Laos, Malaysia, Korea, India, Bangladesh, Qatar, Afghanistan, Vietnam, Nepal, Israel, Oman, Jordan, Taiwan, Bhutan
----------cluster 1----------
Guinea, Liberia, Angola, Madagascar, Gabon, Kenya, Malawi, Eritrea, Somalia, Niger, Mauritania, Gambia, Zimbabwe, Mozambique, Sudan, Mali, Rwanda, Algeria, Uganda, Botswana, Nigeria, Senegal, Zambia, Tunisia, Ghana, Burundi, Namibia
----------cluster 2----------
Portugal, Liechtenstein, Malta, Australia, Germany, Ireland, Finland, Sweden, Japan, England, Iceland, Denmark, Spain, Canada, Norway, Netherlands, Greenland, Belgium, USA, Europe, Italy, Switzerland, Austria, France
----------cluster 3----------
Cyprus, Tajikistan, Azerbaijan, Moldova, Kyrgyzstan, Belarus, Estonia, Turkmenistan, Romania, Ukraine, Hungary, Poland, Russia, Georgia, Serbia, Slovakia, Kazakhstan, Latvia, Albania, Slovenia, Uzbekistan, Armenia, Lithuania, Turkey, Greece, Bulgaria, Croatia, Montenegro, Macedonia
----------cluster 4----------
Venezuela, Brazil, Samoa, Belize, Peru, Argentina, Chile, Suriname, Ecuador, Jamaica, Cuba, Honduras, Nicaragua, Colombia, Philippines, Bahamas, Uruguay, Tuvalu, Guyana, Mexico, Fiji, Dominica
"""
