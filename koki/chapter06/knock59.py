
from sklearn.model_selection import GridSearchCV

params = {'C':[0.01, 0.005, 10]}

gs_model = GridSearchCV(LogisticRegression(max_iter=1500),params, cv=5, verbose=1)
gs_model.fit(x_train, y_train)

best_gs_model = gs_model.best_estimator_
print("\ntrain_score: {:.2%}".format(best_gs_model.score(x_train, y_train)))
print("valid_score: {:.2%}".format(best_gs_model.score(x_valid, y_valid)))
print("test_score: {:.2%}".format(best_gs_model.score(x_test, y_test)))
