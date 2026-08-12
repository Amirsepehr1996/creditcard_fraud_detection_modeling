import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

# import data
data = pd.read_csv(
    r"E:\maktab_sharif\mini_project1_week9\mini-project-01\data\creditcard.csv"
)

# Quick Show
data.info()
print(data.head())
print(data.describe())

# Class Distribution
class_counts = data["Class"].value_counts().sort_index()

plt.figure(figsize=(6, 5))

bars = plt.bar(
    ["Normal (0)", "Fraud (1)"],
    class_counts.values,
    label="Transactions"
)

plt.title("Class Distribution")
plt.xlabel("Transaction Class")
plt.ylabel("Number of Samples")
plt.yscale("log")

for bar in bars:
    height = bar.get_height()
    plt.text(
        bar.get_x() + bar.get_width() / 2,
        height,
        f"{int(height):,}",
        ha="center",
        va="bottom"
    )

plt.legend()
plt.grid(axis="y", linestyle="--", alpha=0.4)
plt.tight_layout()
plt.show()