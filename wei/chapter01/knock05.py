# n-gram

def word_bigram(word_list):
    res = []
    for i in range(len(word_list)-1):
        grams = word_list[i]
        grams += str(' '+ word_list[i+1])
        res.append(grams)
        if len(grams) > 2:
            grams = ''
    return res

def char_bigram(word_str):
    res = []
    #word_str = ''.join(word_list)
    #print(word_str)
    for i in range(len(word_str)-1):
        grams = word_str[i]
        grams += word_str[i+1]
        res.append(grams)
        if len(grams) > 2:
            grams = ''

    return res

if __name__ == '__main__':
    s = 'I am an NLPer'
    words = s.strip().split()
    print(word_bigram(words))
    print(char_bigram(''.join(words)))
