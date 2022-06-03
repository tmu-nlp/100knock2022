import matplotlib.pyplot as plt
import japanize_matplotlib

# cごとの正解率を格納(縦軸)
accuracies_train = []
accuracies_valid = []
accuracies_test = []

# cは正則化パラメータlambdaの逆数
c_list = [0.001, 0.01, 0.1, 1, 10]  # np.linsapce(start, stop, 要素数)

for c in c_list:
    '''モデルの構築, フィッティング'''
    model = LogisticRegression(C = c, random_state = 0)
    model.fit(x_train, y_train)

    '''予測'''
    # 訓練データ
    y_pred_train = model.predict(x_train) 
    accuracy_train = accuracy_score(y_pred_train, y_train)  
    accuracies_train.append(accuracy_train)

    #検証データ
    y_pred_valid = model.predict(x_valid)
    accuracy_valid = accuracy_score(y_pred_valid, y_valid)  
    accuracies_valid.append(accuracy_valid) 

    #テストデータ
    y_pred_test  = model.predict(x_test)
    accuracy_test = accuracy_score(y_pred_test, y_test)  
    accuracies_test.append(accuracy_test) 

    print(f'正則化パラメータ: {c}')
    print(f'正解率(訓練データ): {accuracy_train}')
    print(f'正解率(検証データ): {accuracy_valid}')
    print(f'正解率(テストデータ): {accuracy_test}')
    print('-'*40)

plt.plot(c_list, accuracies_train, label = 'tarin', marker = 'o')
plt.plot(c_list, accuracies_valid, label = 'valid', marker = 'o')
plt.plot(c_list, accuracies_test, label = 'test', marker = 'o')
plt.xlabel('正則化パラメータc')
plt.ylabel('正解率')
plt.legend()
plt.show()
