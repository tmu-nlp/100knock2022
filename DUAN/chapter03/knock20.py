import pandas as pd
 
filename = "jawiki-country.json"
j_data = pd.read_json(filename, lines =True)
df = j_data
uk_df = df[df["title"]=="イギリス"]
uk_df = uk_df["text"].values
print(uk_df)