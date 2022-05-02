
def word_n_gram(target, n):#単語n-gram
    result = []
    target = target.split()
    for i in range(len(target) - n + 1):
        result.append(target[i:i+n])
    return result

def letter_n_gram(target, n):#文字n-gram
    result = []
    for i in range(len(target) - n + 1):
        result.append(target[i:i+n])
    return result

text = 'I am an NLPer'

print(word_n_gram(text,2))
print(letter_n_gram(text,2))
