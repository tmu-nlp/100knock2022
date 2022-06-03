'''
09. Typoglycemia
スペースで区切られた単語列に対して，各単語の先頭と末尾の文字は残し，それ以外の文字の順序をランダムに並び替えるプログラムを作成せよ．
ただし，長さが４以下の単語は並び替えないこととする．
'''

def shuffle_str(s):
    import string
    from random import shuffle

    trantab = s.maketrans(string.punctuation, ' '*len(string.punctuation))
    new_s = s.translate(trantab)
    words = new_s.split()
    res = []

    for word in words:
        if len(word) >= 4:
            chars = list(word[1:-1])
            shuffle(chars)
            new_w = word[0] + ''.join(chars) + word[-1]
        else:
            new_w = word
        res.append(new_w)

    return ' '.join(res)


if __name__ == '__main__':
    sent = 'I couldn’t believe that I could actually understand what I was reading : the phenomenal power of the human mind .'
    print(shuffle_str(sent))