# scripts/baseline.py
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import f1_score

from model.data.labels import encode_labels
from model.preprocessing.clean import clean_text

train = pd.read_csv("model/data/train_split.csv").fillna("")
test = pd.read_csv("model/data/test.csv").fillna("")

X_train = [clean_text(str(t)) for t in train["text"]]
y_train = encode_labels(train)

X_test = [clean_text(str(t)) for t in test["text"]]
y_test = encode_labels(test)

vec = TfidfVectorizer(max_features=10000, ngram_range=(1, 2))
X_train_vec = vec.fit_transform(X_train)
X_test_vec = vec.transform(X_test)

clf = LogisticRegression(max_iter=1000, class_weight="balanced")
clf.fit(X_train_vec, y_train)
preds = clf.predict(X_test_vec)

print(f"TF-IDF + LR F1: {f1_score(y_test, preds):.4f}")