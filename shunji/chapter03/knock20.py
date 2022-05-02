import json 

filename = 'jawiki-country.json'
with open('./jawiki-country.json', 'r') as f:
  for line in f:
    line = json.loads(line)
    if line['title'] == 'イギリス':
      article_uk = line['text']
      break

print(article_uk)
with open('article_uk.txt', 'w') as wf:
    wf.write(article_uk)