from pathlib import Path
from sklearn.datasets import fetch_california_housing
import pandas as pd
import matplotlib.pyplot as plt

# Paths: src/boxplot.py -> project root -> figs/
PROJECT_ROOT = Path(__file__).resolve().parent.parent
FIGS_DIR = PROJECT_ROOT / "figs"
FIGS_DIR.mkdir(exist_ok=True)   # creates figs/ if it doesn't exist

# Load California Housing dataset
housing = fetch_california_housing(as_frame=True)

# Features + target as a single DataFrame
df = housing.frame

# Quick check
print(df.head())
print(df.shape)

# ---- Box plots for every column ----
cols = df.columns
n_cols = 3
n_rows = -(-len(cols) // n_cols)   # ceiling division

fig, axes = plt.subplots(n_rows, n_cols, figsize=(15, 4 * n_rows))
axes = axes.flatten()

for i, col in enumerate(cols):
    axes[i].boxplot(df[col], patch_artist=True,
                    boxprops=dict(facecolor="lightblue"))
    axes[i].set_title(col)
    axes[i].set_xticks([])

# Hide any empty subplots
for j in range(len(cols), len(axes)):
    axes[j].set_visible(False)

plt.suptitle("California Housing - Box Plots", fontsize=16)
plt.tight_layout()

# Save into figs/
out_path = FIGS_DIR / "california_housing_boxplots.png"
plt.savefig(out_path, dpi=300, bbox_inches="tight")
print(f"Saved box plot to: {out_path}")

plt.show()