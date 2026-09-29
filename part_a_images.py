import cv2
import numpy as np
import pandas as pd
from pathlib import Path


crack_dir = Path("data/images/Cracks")
nocrack_dir = Path("data/images/NonCracks")


def extract_features(img_path, canny_low=50, canny_high=150):
    img = cv2.imread(str(img_path))
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

    flat = gray.flatten().astype(np.float32)

    mean_b = float(flat.mean())
    std_b = float(flat.std())
    median_b = float(np.median(flat))
    p25 = float(np.percentile(flat, 25))
    p75 = float(np.percentile(flat, 75))
    dark_ratio = float((flat < 50).mean())
    bright_ratio = float((flat > 200).mean())

    edges = cv2.Canny(gray, canny_low, canny_high)
    edge_density = float((edges > 0).mean())

    return {
        "mean_brightness": mean_b,
        "contrast": std_b,
        "median_brightness": median_b,
        "p25": p25,
        "p75": p75,
        "dark_ratio": dark_ratio,
        "bright_ratio": bright_ratio,
        "edge_density": edge_density,
    }


rows = []
for f in sorted(crack_dir.glob("*.jpg")):
    feats = extract_features(f)
    feats["label"] = 1
    feats["filename"] = f.name
    rows.append(feats)

for f in sorted(nocrack_dir.glob("*.jpg")):
    feats = extract_features(f)
    feats["label"] = 0
    feats["filename"] = f.name
    rows.append(feats)


df = pd.DataFrame(rows)
df.to_csv("data/image_features.csv", index=False)

print("Shape:", df.shape)
print()
print("Feature means by class:")
print(df.groupby("label")[
    ["mean_brightness", "contrast", "dark_ratio",
     "bright_ratio", "edge_density"]
].mean().round(4))
print()
print("Saved to data/image_features.csv")


from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import SVC
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score,
    confusion_matrix,
)
import time
import json


feature_cols = [
    "mean_brightness", "contrast", "median_brightness",
    "p25", "p75", "dark_ratio", "bright_ratio", "edge_density",
]

X = df[feature_cols].values
y = df["label"].values

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y
)

scaler = StandardScaler()
X_train_s = scaler.fit_transform(X_train)
X_test_s  = scaler.transform(X_test)

np.savez("data/image_split.npz",
         X_train=X_train_s, X_test=X_test_s,
         y_train=y_train, y_test=y_test)


def evaluate(name, model, Xtr, Xte):
    t0 = time.perf_counter(); model.fit(Xtr, y_train); tt = time.perf_counter() - t0
    t0 = time.perf_counter(); pred = model.predict(Xte); pt = time.perf_counter() - t0

    print(f"\n[{name}]")
    print(f"  Train time: {tt:.4f}s  Pred time: {pt:.4f}s")
    print(f"  Accuracy:  {accuracy_score(y_test, pred):.4f}")
    print(f"  Precision: {precision_score(y_test, pred):.4f}")
    print(f"  Recall:    {recall_score(y_test, pred):.4f}")
    print(f"  F1:        {f1_score(y_test, pred):.4f}")
    print(f"  Confusion matrix:\n{confusion_matrix(y_test, pred)}")

    return {
        "name": name,
        "train_time": tt, "pred_time": pt,
        "accuracy":  accuracy_score(y_test, pred),
        "precision": precision_score(y_test, pred),
        "recall":    recall_score(y_test, pred),
        "f1":        f1_score(y_test, pred),
    }


print("\n" + "=" * 60)
print("IMAGE CLASSIFIERS")
print("=" * 60)

results = []
results.append(evaluate("LogReg", LogisticRegression(max_iter=1000), X_train_s, X_test_s))
results.append(evaluate("RandomForest", RandomForestClassifier(n_estimators=100, random_state=42), X_train_s, X_test_s))
results.append(evaluate("SVM-RBF", SVC(kernel="rbf"), X_train_s, X_test_s))

with open("data/part_a_results.json", "w") as f:
    json.dump(results, f, indent=2)

print("\nSaved to data/part_a_results.json")