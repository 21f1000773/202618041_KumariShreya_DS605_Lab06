

import cv2
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path

crack_dir = Path("data/images/Cracks")
nocrack_dir = Path("data/images/NonCracks")

crack_sample = sorted(crack_dir.glob("*.jpg"))[0]
nocrack_sample = sorted(nocrack_dir.glob("*.jpg"))[0]

fig, axes = plt.subplots(2, 3, figsize=(12, 8))

for row, (img_path, label) in enumerate([
    (crack_sample, "Crack"),
    (nocrack_sample, "NonCrack"),
]):
    img = cv2.imread(str(img_path))
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

    rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    axes[row, 0].imshow(rgb)
    axes[row, 0].set_title(f"{label} — original")
    axes[row, 0].axis("off")

    axes[row, 1].imshow(gray, cmap="gray")
    axes[row, 1].set_title(f"{label} — grayscale")
    axes[row, 1].axis("off")

    edges_50 = cv2.Canny(gray, 50, 150)
    axes[row, 2].imshow(edges_50, cmap="gray")
    axes[row, 2].set_title(f"{label} — Canny (50, 150)")
    axes[row, 2].axis("off")

plt.tight_layout()
plt.savefig("data/sample_visualization.png", dpi=100, bbox_inches="tight")
print("Saved to data/sample_visualization.png")