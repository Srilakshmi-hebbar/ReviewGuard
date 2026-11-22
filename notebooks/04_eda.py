import os
import pandas as pd
import numpy as np
import matplotlib
matplotlib.use("Agg")  # draw to files, no pop-up windows
import matplotlib.pyplot as plt

plt.style.use("dark_background")
C0, C1 = "#00c2ff", "#ff4d6d"   # class 0 = cyan, class 1 = red
os.makedirs("reports/eda", exist_ok=True)

train = pd.read_csv("data/train_clean.csv", sep="\t",
                    parse_dates=["date", "yelpJoinDate"])
train["account_age_days"] = (train["date"] - train["yelpJoinDate"]).dt.days

c0 = train[train["flagged"] == 0]
c1 = train[train["flagged"] == 1]

# 1) Class balance
fig, ax = plt.subplots(figsize=(5, 4))
counts = train["flagged"].value_counts().sort_index()
ax.bar(["0", "1"], counts.values, color=[C0, C1])
for i, v in enumerate(counts.values):
    ax.text(i, v + 50, str(v), ha="center")
ax.set_title("Class balance (flagged)")
ax.set_ylabel("Reviews")
fig.tight_layout()
fig.savefig("reports/eda/1_class_balance.png", dpi=150)
plt.close(fig)

# 2) Reviewer profile features, log scale
fig, axes = plt.subplots(2, 2, figsize=(11, 8))
for ax, col in zip(axes.ravel(),
                   ["reviewCount", "friendCount", "fanCount", "usefulCount"]):
    bins = np.linspace(0, np.log1p(train[col]).max(), 30)
    ax.hist(np.log1p(c0[col]), bins=bins, alpha=0.65, color=C0, label="0")
    ax.hist(np.log1p(c1[col]), bins=bins, alpha=0.65, color=C1, label="1")
    ax.set_title(f"{col} (log scale)")
    ax.set_xlabel(f"log(1 + {col})")
    ax.legend()
fig.tight_layout()
fig.savefig("reports/eda/2_profile_features.png", dpi=150)
plt.close(fig)

# 3) Account age and review length
fig, axes = plt.subplots(1, 2, figsize=(11, 4))
for ax, col, title in zip(
    axes, ["account_age_days", "ReviewLength"],
    ["Account age at review (days)", "Review length (words, cleaned text)"]):
    bins = np.linspace(0, np.log1p(train[col]).max(), 30)
    ax.hist(np.log1p(c0[col]), bins=bins, alpha=0.65, color=C0, label="0")
    ax.hist(np.log1p(c1[col]), bins=bins, alpha=0.65, color=C1, label="1")
    ax.set_title(title + ", log scale")
    ax.set_xlabel(f"log(1 + {col})")
    ax.legend()
fig.tight_layout()
fig.savefig("reports/eda/3_age_and_length.png", dpi=150)
plt.close(fig)

# 4) Rating distribution by class (as shares, so classes are comparable)
fig, ax = plt.subplots(figsize=(6, 4))
share = pd.crosstab(train["rating"], train["flagged"], normalize="columns")
x = np.arange(len(share.index))
ax.bar(x - 0.2, share[0], width=0.4, color=C0, label="0")
ax.bar(x + 0.2, share[1], width=0.4, color=C1, label="1")
ax.set_xticks(x)
ax.set_xticklabels(share.index)
ax.set_title("Star rating share within each class")
ax.set_xlabel("Rating")
ax.set_ylabel("Share of class")
ax.legend()
fig.tight_layout()
fig.savefig("reports/eda/4_rating_share.png", dpi=150)
plt.close(fig)

# 5) Correlation of each numeric feature with the label
num_cols = ["rating", "reviewUsefulCount", "friendCount", "reviewCount",
            "firstCount", "usefulCount", "coolCount", "funnyCount",
            "complimentCount", "tipCount", "fanCount", "restaurantRating",
            "ReviewLength", "account_age_days"]
corr = train[num_cols + ["flagged"]].corr(method="spearman")["flagged"]
corr = corr.drop("flagged").sort_values()
fig, ax = plt.subplots(figsize=(7, 6))
ax.barh(corr.index, corr.values, color=[C1 if v > 0 else C0 for v in corr.values])
ax.set_title("Spearman correlation with flagged")
fig.tight_layout()
fig.savefig("reports/eda/5_correlation_with_label.png", dpi=150)
plt.close(fig)

# Text summary
print("=== Spearman correlation with flagged ===")
print(corr.round(3))
print("\n=== Median account age (days) by class ===")
print(train.groupby("flagged")["account_age_days"].median())
print("\n=== Rating share by class ===")
print(share.round(3))
print("\n=== Share of reviews written within 30 days of joining ===")
print((train["account_age_days"] <= 30).groupby(train["flagged"]).mean().round(3))
print("\nCharts saved in reports/eda")