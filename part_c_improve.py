

import cv2
import numpy as np
import pandas as pd
from pathlib import Path
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import SVC
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
import time
import json


crack_dir = Path("data/images/Cracks")
nocrack_dir = Path("data/images/NonCracks")


def extract_improved(img_path, canny_low, canny_high):
    img = cv2.imread(str(img_path))
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY).astype(np.float32)

    mean_b = float(gray.mean())
    std_b = float(gray.std())
    dark_ratio = float((gray < 50).mean())
    bright_ratio = float((gray > 200).mean())

    edges = cv2.Canny(gray.astype(np.uint8), canny_low, canny_high)
    edge_density = float((edges > 0).mean())

    return {
        "mean_brightness": mean_b,
        "contrast": std_b,
        "dark_ratio": dark_ratio,
        "bright_ratio": bright_ratio,
        "edge_density": edge_density,
    }


def build_features(canny_low, canny_high):
    rows = []
    for f in sorted(crack_dir.glob("*.jpg")):
        r = extract_improved(f, canny_low, canny_high)
        r["label"] = 1
        rows.append(r)
    for f in sorted(nocrack_dir.glob("*.jpg")):
        r = extract_improved(f, canny_low, canny_high)
        r["label"] = 0
        rows.append(r)
    return pd.DataFrame(rows)


print("Testing Canny thresholds")
for low, high in [(50, 150), (100, 200), (150, 250), (200, 300)]:
    df_try = build_features(low, high)
    ed_crack = df_try[df_try.label == 1]["edge_density"].mean()
    ed_non = df_try[df_try.label == 0]["edge_density"].mean()
    gap = abs(ed_crack - ed_non)
    print(f"  low={low} high={high}  crack={ed_crack:.4f}  non={ed_non:.4f}  gap={gap:.4f}")


BEST_LOW, BEST_HIGH = 50, 150
df = build_features(BEST_LOW, BEST_HIGH)
df.to_csv("data/image_features_improved.csv", index=False)

print()
print("Improved feature means by class:")
print(df.groupby("label")[
    ["mean_brightness", "contrast", "dark_ratio", "bright_ratio", "edge_density"]
].mean().round(4))


feature_cols = [
    "mean_brightness", "contrast", "dark_ratio", "bright_ratio", "edge_density",
]
X = df[feature_cols].values
y = df["label"].values

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y
)
scaler = StandardScaler()
X_train_s = scaler.fit_transform(X_train)
X_test_s = scaler.transform(X_test)


def evaluate(name, model):
    t0 = time.perf_counter(); model.fit(X_train_s, y_train); tt = time.perf_counter() - t0
    t0 = time.perf_counter(); pred = model.predict(X_test_s); pt = time.perf_counter() - t0
    return {
        "model": name,
        "train_time": tt, "pred_time": pt,
        "accuracy": accuracy_score(y_test, pred),
        "precision": precision_score(y_test, pred),
        "recall": recall_score(y_test, pred),
        "f1": f1_score(y_test, pred),
    }


print("\n" + "=" * 60)
print("IMPROVED IMAGE CLASSIFIERS")
print("=" * 60)

results = []
for name, model in [
    ("LogReg", LogisticRegression(max_iter=1000)),
    ("RandomForest", RandomForestClassifier(n_estimators=300, max_depth=None, class_weight="balanced", random_state=42)),
    ("SVM-RBF", SVC(kernel="rbf")),
]:
    r = evaluate(name, model)
    results.append(r)
    print(f"\n[{name}]")
    print(f"  Train time: {r['train_time']:.4f}s  Pred time: {r['pred_time']:.4f}s")
    print(f"  Accuracy:  {r['accuracy']:.4f}")
    print(f"  Precision: {r['precision']:.4f}")
    print(f"  Recall:    {r['recall']:.4f}")
    print(f"  F1:        {r['f1']:.4f}")

with open("data/part_c_results.json", "w") as f:
    json.dump(results, f, indent=2)

print("\nSaved to data/part_c_results.json")