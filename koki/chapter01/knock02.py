str1 = 'パトカー'
str2 = 'タクシー'
new_str = ''

for (char1, char2) in zip(str1, str2):#zip() 複数のリストをまとめる
    new_str += char1 + char2
print(new_str)
