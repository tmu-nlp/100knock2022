import json

with open("jawiki-country.json", "r") as my_file:
    with open("out.txt", "w") as out:
        for line in my_file:
            my_json = json.loads(line)
            if my_json['title'] in "イギリス":
                print(my_json['text'])