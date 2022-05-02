#100本ノック第1章05
#与えられたシーケンス（文字列やリストなど）からn-gramを作る関数を作成せよ．
# この関数を用い，”I am an NLPer”という文から単語bi-gram，文字bi-gramを得よ．

#line = "I am an NLPer"

def char_ngram(line,n):#文字bi-gram
    result_char = []

    for i in range(len(line)-n+1):
        result_char.append(line[i:i+n])

    return(result_char)

def word_ngram(line,n):#単語bi-gram
    word_list = line.split()#単語は空白区切りでリストに格納
    result_word = []

    for i in range(len(word_list)-n+1):
        result_word.append(word_list[i:i+n])

    return(result_word)        

print("文字bi-gram:",char_ngram("I am an NLPer",2))
print("単語bi-gram:",word_ngram("I am an NLPer",2))



