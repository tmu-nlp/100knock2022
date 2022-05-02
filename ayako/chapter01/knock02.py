#100本ノック第1章02
#「パトカー」＋「タクシー」の文字を先頭から交互に連結して文字列「パタトクカシーー」を得よ．

str1 = "パトカー"
str2 = "タクシー"
result = ""

for char in range(len(str1)):
    result += str1[char]
    result += str2[char] 

print(result)
