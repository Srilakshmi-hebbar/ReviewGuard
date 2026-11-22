import os, sys
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import GroupKFold, cross_val_score
from sklearn.metrics import accuracy_score, f1_score
from src.features import build_features

dates = ["date", "yelpJoinDate"]
train = pd.read_csv("data/train_clean.csv", sep="\t", parse_dates=dates)
test = pd.read_csv("data/test_clean.csv", sep="\t", parse_dates=dates)

y_train, y_test = train["flagged"], test["flagged"]
groups = train["reviewerID"]
unseen = ~test["reviewerID"].isin(set(train["reviewerID"]))

X_train = build_features(train)
X_test = build_features(test)

variants = {
    "WITHOUT reviewUsefulCount (our features)": (X_train, X_test),
    "WITH reviewUsefulCount": (
        X_train.assign(reviewUsefulCount=train["reviewUsefulCount"].values),
        X_test.assign(reviewUsefulCount=test["reviewUsefulCount"].values),
    ),
}

rows = []
for name, (Xtr, Xte) in variants.items():
    rf = RandomForestClassifier(n_estimators=300, random_state=42, n_jobs=-1)
    cv = cross_val_score(rf, Xtr, y_train, groups=groups,
                         cv=GroupKFold(n_splits=5), scoring="f1_macro")
    rf.fit(Xtr, y_train)
    pred = rf.predict(Xte)
    rows.append({
        "variant": name,
        "groupCV_macroF1": cv.mean(),
        "test_acc": accuracy_score(y_test, pred),
        "test_macroF1": f1_score(y_test, pred, average="macro"),
        "unseenReviewers_macroF1": f1_score(y_test[unseen], pred[unseen],
                                            average="macro"),
    })

print("=== RANDOM FOREST: effect of reviewUsefulCount ===")
print("(Your original result: accuracy 0.8913, macro F1 0.8910)")
print(pd.DataFrame(rows).set_index("variant").round(4).T)

print("\n=== RESTAURANT CHECK ===")
seen_rest = set(train["restaurantID"])
print("Unique restaurants in train:", train["restaurantID"].nunique())
print("Unique restaurants in test: ", test["restaurantID"].nunique())
print("Share of test rows from restaurants also in train:",
      round(float(test["restaurantID"].isin(seen_rest).mean()), 3))

per_rest = train.groupby("restaurantID")["flagged"].agg(["count", "mean"])
big = per_rest[per_rest["count"] >= 10]
print("\nRestaurants with 10+ train reviews:", len(big))
if len(big) > 0:
    print("Share of those where one class is 80%+ of reviews:",
          round(float(((big["mean"] <= 0.2) | (big["mean"] >= 0.8)).mean()), 3))
    print("Label rate (mean of flagged) across those restaurants:")
    print(big["mean"].describe().round(3))