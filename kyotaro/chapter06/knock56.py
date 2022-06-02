"""
52で学習したロジスティック回帰モデルの適合率、再現率、F1スコアを、評価データ上で計測せよ。
カテゴリごとに適合率、再現率、F1スコアを求め、カテゴリごとの性能をマイクロ平均（micro-average）とマクロ平均（macro-average）で統合せよ
"""
from sklearn.metrics import precision_score, recall_score, f1_score, classification_report
from knock54 import train_ans, test_ans, train_myans, test_myans

model_performance = classification_report(train_ans, train_myans)
print(model_performance)
