

import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfTransformer
from sklearn.linear_model import LogisticRegression
from sklearn.naive_bayes import MultinomialNB
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score, confusion_matrix,
)
import time
import json


df = pd.read_csv("data/emails.csv")

y = df["Prediction"].values
X_counts = df.drop(columns=["Email No.", "Prediction"]).values.astype(np.float32)

print("Counts shape:", X_counts.shape)
print("Labels:", dict(zip(*np.unique(y, return_counts=True))))
print()


X_train, X_test, y_train, y_test = train_test_split(
    X_counts, y, test_size=0.2, random_state=42, stratify=y
)

# TF-IDF reweighting on top of the raw counts
tfidf = TfidfTransformer()
X_train_tfidf = tfidf.fit_transform(X_train)
X_test_tfidf  = tfidf.transform(X_test)


def evaluate(name, model, Xtr, Xte):
    t0 = time.perf_counter(); model.fit(Xtr, y_train); tt = time.perf_counter() - t0
    t0 = time.perf_counter(); pred = model.predict(Xte); pt = time.perf_counter() - t0

    cm = confusion_matrix(y_test, pred)
    print(f"\n[{name}]")
    print(f"  Train time: {tt:.4f}s  Pred time: {pt:.4f}s")
    print(f"  Accuracy:  {accuracy_score(y_test, pred):.4f}")
    print(f"  Precision: {precision_score(y_test, pred):.4f}")
    print(f"  Recall:    {recall_score(y_test, pred):.4f}")
    print(f"  F1:        {f1_score(y_test, pred):.4f}")
    print(f"  Confusion matrix:\n{cm}")

    return {
        "model": name,
        "features": Xtr.shape[1],
        "train_time": tt, "pred_time": pt,
        "accuracy":  accuracy_score(y_test, pred),
        "precision": precision_score(y_test, pred),
        "recall":    recall_score(y_test, pred),
        "f1":        f1_score(y_test, pred),
    }


print("=" * 60)
print("COUNT REPRESENTATION")
print("=" * 60)
results = []
results.append(evaluate("LogReg on counts",  LogisticRegression(max_iter=1000), X_train, X_test))
results.append(evaluate("MNB on counts",     MultinomialNB(), X_train, X_test))

print("\n" + "=" * 60)
print("TF-IDF REPRESENTATION")
print("=" * 60)
results.append(evaluate("LogReg on tfidf",  LogisticRegression(max_iter=1000), X_train_tfidf, X_test_tfidf))
results.append(evaluate("MNB on tfidf",     MultinomialNB(), X_train_tfidf, X_test_tfidf))


with open("data/part_b_results.json", "w") as f:
    json.dump(results, f, indent=2)

print("\nSaved to data/part_b_results.json")