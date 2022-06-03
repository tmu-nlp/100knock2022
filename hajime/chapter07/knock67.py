from gensim.models import KeyedVectors
from collections import defaultdict
from sklearn.cluster import KMeans
import re
import pickle
import numpy as np

# 国名に関する単語ベクトル
# http://www.fao.org/countryprofiles/iso3list/en/

with open("word2vec.pkl", "rb") as f:
    model = pickle.load(f)

country_set = set()

with open("country.csv", "r") as f:
    next(f)
    for line in f:
        line_list = line.strip().split(",")
        trg_word = line_list[1]
        trg_word = re.sub("\(.*\)", "", trg_word)
        trg_word = re.sub(" ", "_", trg_word)
        trg_word = trg_word.rstrip("_")
        country_set.add(trg_word)

country_list = list(country_set)
country_list_in = []
country_vec = []

for country in country_list:
    if country in model.index_to_key:
        country_vec.append(model[country])
        country_list_in.append(country)

kmeans = KMeans(n_clusters=5, random_state=20010101)
clus_num = kmeans.fit_predict(country_vec)

if __name__ == "__main__":
    for i in range(5):
        cluster = np.where(kmeans.labels_ == i)[0]
        print('cluster', i)
        print(', '.join([country_list_in[k] for k in cluster]))


"""
cluster 0
Argentina, Barbados, Haiti, Panama, Venezuela, Bolivia, Martinique, Suriname, Belize, Nicaragua, Chile, Mexico, Canada, Costa_Rica, Puerto_Rico, Paraguay, Dominica, Guadeloupe, Ecuador, Colombia, Dominican_Republic, Honduras, Uruguay, Jamaica, Aruba, Brazil, Cuba, El_Salvador, Peru, French_Guiana, Guatemala, Grenada, Guyana, Philippines
cluster 1
Namibia, Congo, Equatorial_Guinea, Mauritania, Algeria, Mali, Lesotho, Madagascar, Togo, Zambia, Ethiopia, Malawi, Senegal, Benin, Djibouti, Ghana, Gabon, Sudan, Comoros, Botswana, Cameroon, Sierra_Leone, Zimbabwe, Mozambique, Nigeria, Angola, Tunisia, Rwanda, Liberia, Somalia, Kenya, Burkina_Faso, Uganda, Guinea, South_Africa, Burundi, Eritrea, Chad, Western_Sahara
cluster 2
Persian_Gulf, Uzbekistan, Lebanon, Occupied_Palestinian_Territory, Brunei_Darussalam, Afghanistan, Pakistan, Thailand, Australia, Morocco, Israel, India, Tajikistan, United_Arab_Emirates, Cambodia, China, Kyrgyzstan, Sri_Lanka, Kuwait, Jordan, Bahrain, Iran, Malaysia, Turkmenistan, Saudi_Arabia, Iraq, Oman, Bangladesh, Mongolia, Bhutan, Indonesia, Qatar, Japan, Palestine, Gaza_Strip, Singapore, Arunachal_Pradesh, Viet_Nam, Nepal, Myanmar, Holy_See, Libya, Egypt, Yemen, Golan_Heights
cluster 3
Spain, Turkey, Germany, Kazakhstan, Estonia, Norway, Hungary, Switzerland, Iceland, Bulgaria, Luxembourg, Liechtenstein, Georgia, Romania, Denmark, Kosovo, Andorra, Lithuania, Netherlands, Portugal, Poland, Czechia, Finland, Ireland, Azerbaijan, Latvia, Italy, Croatia, Belarus, Serbia, Malta, European_Union, Sweden, San_Marino, Belgium, Slovakia, France, Montenegro, Cyprus, Austria, Armenia, Slovenia, Ukraine, Greece, Faroe_Islands, Monaco, Albania
cluster 4
Greater_Antilles, Channel_Islands, Nauru, Solomon_Islands, Kingman_Reef, Seychelles, Clipperton_Island, Liancourt_Rocks, Paracel_Islands, Kerguelen_Islands, Niue, Fiji, Åland_Islands, Guam, Antarctica, Pitcairn, Cayman_Islands, Lesser_Antilles, Micronesia, British_Virgin_Islands, Bouvet_Island, Palmyra_Atoll, Palau, Greenland, Samoa, Saint_Lucia, Bermuda, Vanuatu, Curaçao, Maldives, Mayotte, Cabo_Verde, French_Polynesia, Tokelau, Montserrat, Madeira_Islands, Northern_Mariana_Islands, Marshall_Islands, Jersey, Spratly_Islands, Mauritius, New_Zealand, Senkaku_Islands, Réunion, Sint_Maarten, Falkland_Islands, Guernsey, Gibraltar, Kuril_Islands, Anguilla, Canary_Islands, Tuvalu, Saint_Barthélemy, Cocos, Kiribati, Cook_Islands, Johnston_Atoll, Tonga, Chagos_Archipelago, Diego_Garcia
"""
