import random
text = 'I couldn’t believe that I could actually understand what I was reading : the phenomenal power of the human mind .'

def Typoglycemia(trg):
    l_word = []
    l_word = trg.split()

    l_result = []
    result = ''
    
    for idx, word in enumerate(l_word):
        if len(word) > 4:
            trg_shuffle = list(word[1:-1])
            l_result.append(word[0] + ''.join(random.sample(trg_shuffle, len(trg_shuffle))) + word[-1])
            #l_result.append(word[0] + ''.join(random.shuffle(trg_shuffle)) + word[-1])
        else:
            l_result.append(word)

    result = ' '.join(l_result)
    return result


print(Typoglycemia(text))
