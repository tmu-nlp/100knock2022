from sklearn.metrics import precision_score, recall_score, f1_score

def metrics(y_data, y_pred, ave=None):
  precision_sco = precision_score(y_data, y_pred, average=ave)
  recall_sco = recall_score(y_data, y_pred, average=ave)
  f1_sco = f1_score(y_data, y_pred, average=ave)
  form = "適合率：{}\n再現率 : {}\nF1 : {}\n".format(precision_sco, recall_sco, f1_sco)
  return form

print(f"【カテゴリ順】{model.classes_}\n\n{metrics(y_test, y_test_pred)}")
print("【マクロ平均】\n", metrics(y_test, y_test_pred, "macro"))
print("【マイクロ平均】\n", metrics(y_test, y_test_pred, "micro"))
