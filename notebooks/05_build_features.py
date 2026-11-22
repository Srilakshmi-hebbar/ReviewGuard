import os
import sys

# Let this script import from the project root (so "src" is found)
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

import numpy as np
import pandas as pd
from src.features import build_features, BEHAVIOURAL_FEATURES

pd.set_option("display.width", 200)

dates = ["date", "yelpJoinDate"]
train = pd.read_csv("data/train_clean.csv", sep="\t", parse_dates=dates)
test = pd.read_csv("data/test_clean.csv", sep="\t", parse_dates=dates)

X_train, y_train = build_features(train), train["flagged"]
X_test, y_test = build_features(test), test["flagged"]

print("X_train:", X_train.shape, " X_test:", X_test.shape)
print("Number of features:", len(BEHAVIOURAL_FEATURES))
print("Same columns in train and test:",
      list(X_train.columns) == list(X_test.columns))

print("\nMissing values (train/test):",
      int(X_train.isna().sum().sum()), "/", int(X_test.isna().sum().sum()))
print("Infinite values (train/test):",
      int(np.isinf(X_train.to_numpy(dtype=float)).sum()), "/",
      int(np.isinf(X_test.to_numpy(dtype=float)).sum()))

print("\n'reviewUsefulCount' in features?",
      "reviewUsefulCount" in X_train.columns)
print("'flagged' in features?", "flagged" in X_train.columns)

print("\n=== Engineered feature means by class (train) ===")
engineered = ["account_age_days", "reviewed_within_30d", "is_extreme_rating",
              "rating_gap", "friends_per_review", "useful_per_review",
              "compliments_per_review", "no_social_ties"]
print(X_train[engineered].groupby(y_train).mean().round(3).T)

print("\n=== First 3 rows ===")
print(X_train.head(3).T)