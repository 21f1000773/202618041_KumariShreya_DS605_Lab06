import json
import pandas as pd


with open("data/part_a_results.json") as f:
    a = json.load(f)
with open("data/part_b_results.json") as f:
    b = json.load(f)
with open("data/part_c_results.json") as f:
    c = json.load(f)


print("=" * 80)
print("IMAGE CLASSIFICATION: PART A vs PART C")
print("=" * 80)

image_rows = []
for r in a:
    image_rows.append(["A", r["name"], f"{r['train_time']:.4f}", f"{r['pred_time']:.4f}",
                       f"{r['accuracy']:.4f}", f"{r['precision']:.4f}",
                       f"{r['recall']:.4f}", f"{r['f1']:.4f}"])
for r in c:
    image_rows.append(["C", r["model"], f"{r['train_time']:.4f}", f"{r['pred_time']:.4f}",
                       f"{r['accuracy']:.4f}", f"{r['precision']:.4f}",
                       f"{r['recall']:.4f}", f"{r['f1']:.4f}"])

image_df = pd.DataFrame(image_rows, columns=[
    "Part", "Model", "Train (s)", "Pred (s)", "Accuracy", "Precision", "Recall", "F1"
])
print(image_df.to_string(index=False))


print()
print("=" * 80)
print("TEXT CLASSIFICATION: COUNTS vs TF-IDF")
print("=" * 80)

text_df = pd.DataFrame([
    [r["model"], r["features"], f"{r['train_time']:.4f}", f"{r['pred_time']:.4f}",
     f"{r['accuracy']:.4f}", f"{r['precision']:.4f}",
     f"{r['recall']:.4f}", f"{r['f1']:.4f}"]
    for r in b
], columns=["Model", "Features", "Train (s)", "Pred (s)",
            "Accuracy", "Precision", "Recall", "F1"])
print(text_df.to_string(index=False))


with open("data/comparison.md", "w") as f:
    f.write("# Lab 6 Comparison Tables\n\n")
    f.write("## Image Classification: Part A vs Part C\n\n")
    f.write(image_df.to_markdown(index=False))
    f.write("\n\n## Text Classification: Counts vs TF-IDF\n\n")
    f.write(text_df.to_markdown(index=False))

print()
print("Saved to data/comparison.md")