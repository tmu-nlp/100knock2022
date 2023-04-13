str = 'パタトクカシーー'
new_str = ''
for idx, char in enumerate(str):
    if (idx + 1) % 2 == 1:#奇数番目のみ取り出す
       new_str += char
    else:
        pass
print(new_str)
