#100本ノック第1章03
#“Now I need a drink, alcoholic of course, after the heavy lectures involving quantum mechanics.”
# という文を単語に分解し，各単語の（アルファベットの）文字数を先頭から出現順に並べたリストを作成せよ．

line = "Now I need a drink, alcoholic of course, after the heavy lectures involving quantum mechanics."

new_line = line.replace(",","").replace(".","")#カンマを除く
my_list = new_line.split()#文を空白区切りで配列に格納

num_list = []

for value in my_list:
    num_list.append(len(value))

print(num_list)




