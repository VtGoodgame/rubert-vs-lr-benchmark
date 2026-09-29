# scripts/baseline.py
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import f1_score
import pandas as pd
from model.preprocessing.clean import clean_text

train = pd.read_csv("model/data/train_split.csv").fillna("")
test = pd.read_csv("model/data/test.csv").fillna("")

X_train = [clean_text(str(t)) for t in train["text"]]
y_train = train["target"].map({"ham": 0, "spam": 1}).values

X_test = [clean_text(str(t)) for t in test["text"]]
y_test = test["target"].map({"ham": 0, "spam": 1}).values

vec = TfidfVectorizer(max_features=10000, ngram_range=(1, 2))
X_train_vec = vec.fit_transform(X_train)
X_test_vec = vec.transform(X_test)

clf = LogisticRegression(max_iter=1000, class_weight="balanced")
clf.fit(X_train_vec, y_train)
preds = clf.predict(X_test_vec)

print(f"TF-IDF + LR F1: {f1_score(y_test, preds):.4f}")