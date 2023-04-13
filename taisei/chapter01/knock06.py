#006
# -*- coding : utf-8 -*-
import numpy as np

def ngram_letter(n, sen):
    ngram_list = []
    for i in range (len(sen) - n + 1):
        str = ''
        for j in range (n):
            str = str + sen[i + j]
        ngram_list.append(str)
    return ngram_list

# def do_unique(lis): #集合だから重複要素をなくすための関数
#     lis_new = []
#     for i in range (len(lis)):
#         same_flag = 0
#         for j in range (i + 1, len(lis)):
#             if (lis[i] == lis[j]):
#                 same_flag = 1
#         if (same_flag == 0):
#             lis_new.append(lis[i])
#     return lis_new

str6 = 'paraparaparadise'
str7 = 'paragraph'
x = ngram_letter(2, str6)
y = ngram_letter(2, str7)
# x = do_unique(x)
# y = do_unique(y)
x = np.unique(x)
y = np.unique(y)

print("X : ", x)
print("Y : ", y)
#和集合
x_or_y = y.copy()
for i in range (x.shape[0]):
    if (x[i] in y):
        pass
    else:
        x_or_y = np.append(x_or_y , x[i])
print('和：' , x_or_y)

#積集合
x_and_y = np.array([])
for i in range (x.shape[0]):
    if (x[i] in y):
        x_and_y = np.append(x_and_y, x[i])
print('積：' , x_and_y)

#差集合
x_minus_y = np.array([])
for i in range (x_or_y.shape[0]):
    if (x_or_y[i] in y):
        pass
    else:
        x_minus_y = np.append(x_minus_y, x_or_y[i])

y_minus_x = np.array([])
for i in range (x_or_y.shape[0]):
    if (x_or_y[i] in x):
        pass
    else:
        y_minus_x = np.append(y_minus_x, x_or_y[i])

print('差(x-y)：', x_minus_y)
print('差(y-x)：', y_minus_x)
