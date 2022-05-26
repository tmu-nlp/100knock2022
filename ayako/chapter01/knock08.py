#100本ノック第1章08
#与えられた文字列の各文字を，以下の仕様で変換する関数cipherを実装せよ．
# ・英小文字ならば(219 - 文字コード)の文字に置換
# ・その他の文字はそのまま出力
#この関数を用い，英語のメッセージを暗号化・復号化せよ．


def cipher(line):
    my_list = list(line)
    new_line = ""
    for char in my_list:
        if char.islower():#英小文字の場合
            new_char = chr(219-ord(char))
            new_line += new_char
        else:
            new_line += char

    return(new_line)

print(cipher("I Am An Engineer."))
print(cipher(cipher("I Am An Engineer.")))