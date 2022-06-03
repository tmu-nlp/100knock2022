'''
題目:　
関数cipherで、英小文字ならば(219‐文字コード)の文字に置換せよ。
その他の文字はそのまま出力
ord()関数:　英文字のascii値(0~255)を返す.
chr()関数:　ascii値から英文字に戻す
'''
def cipher(word_str):
    ascii_values = list(range(ord('a'),ord('z')+1))
    new_str = ''
    for i in range(len(word_str)):
        if ord(word_str[i]) in ascii_values:
            new_word = word_str[i].replace(word_str[i], chr(219 - ord(word_str[i])))
            new_str += new_word
        else:
            char = word_str[i]
            new_str += char
    return new_str

if __name__ == '__main__':
    str ='I have a dream. Tomorrow is another day. スーパーマーケット。今天天气真好呀~'
    print(cipher(str))

