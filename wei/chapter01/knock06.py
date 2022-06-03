from knock05 import char_bigram

def is_in(bigrams):
    if 'se' in bigrams:
        return True
    else:
        return False

if __name__ == '__main__':
    x = 'paraparaparadise'
    y = 'paragraph'
    X = set(char_bigram(x))
    Y = set(char_bigram(y))
    print(X)
    print(Y)
    print('X+Y=', X | Y)
    print('X&Y=', X & Y)
    print('X-Y=', X-Y)
    print('Y-X=', Y-X)
    print(is_in(X))
    print(is_in(Y))