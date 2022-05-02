def n_gram(sen,n,CorW):
    if CorW == 'W':
        sen = sen.split(' ')
    elif CorW == 'C':
        sen = sen
    else:
        print('n_gram(sen,n,CorW)')
        print('CorW:Plsease Write C (Character) or W (Word)')
    x_list = list()
    for i in range(len(sen)-n+1):
        x_list.append(sen[i:i+n])
    return(x_list)
a = 'I am an NLPer'
print(n_gram(a,2,'C'))
#['I ', ' a', 'am', 'm ', ' a', 'an', 'n ', ' N', 'NL', 'LP', 'Pe', 'er']
print(n_gram(a,2,'W'))
#[['I', 'am'], ['am', 'an'], ['an', 'NLPer']]