y_pred = model.predict(x_valid)
print(y_valid.values)
print(y_pred)

y_pred = model.predict_proba(x_valid)
print(y_pred)
