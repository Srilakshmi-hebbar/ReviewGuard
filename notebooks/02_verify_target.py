import pandas as pd

pd.set_option("display.max_columns", None)
pd.set_option("display.width", 200)

train = pd.read_csv("data/new_data_train.csv", sep="\t")
test = pd.read_csv("data/new_data_test.csv", sep="\t")

behaviour_cols = [
    "rating", "reviewUsefulCount", "friendCount", "reviewCount",
    "firstCount", "usefulCount", "coolCount", "funnyCount",
    "complimentCount", "tipCount", "fanCount",
    "restaurantRating", "ReviewLength",
]

print("=== MEAN OF EACH FEATURE BY flagged CLASS ===")
print(train.groupby("flagged")[behaviour_cols].mean().round(2).T)

print("\n=== MEDIAN OF EACH FEATURE BY flagged CLASS ===")
print(train.groupby("flagged")[behaviour_cols].median().T)

print("\n=== SHARE OF 'THIN PROFILE' REVIEWERS BY CLASS ===")
thin = pd.DataFrame({
    "reviewCount_le_1": train["reviewCount"].le(1),
    "friendCount_eq_0": train["friendCount"].eq(0),
    "fanCount_eq_0": train["fanCount"].eq(0),
    "flagged": train["flagged"],
})
print(thin.groupby("flagged").mean().round(3))

print("\n=== REVIEWER OVERLAP (matters for reviewer-independent validation) ===")
print("Unique reviewers in train:", train["reviewerID"].nunique())
print("Unique reviewers in test: ", test["reviewerID"].nunique())
shared = set(train["reviewerID"]) & set(test["reviewerID"])
print("Reviewers appearing in BOTH train and test:", len(shared))
labels_per_reviewer = train.groupby("reviewerID")["flagged"].nunique()
print("Train reviewers with mixed labels (both 0 and 1):",
      int((labels_per_reviewer > 1).sum()))

print("\n=== DATE RANGES (matters for temporal leakage) ===")
train["date_parsed"] = pd.to_datetime(train["date"], format="%m/%d/%Y", errors="coerce")
test["date_parsed"] = pd.to_datetime(test["date"], format="%m/%d/%Y", errors="coerce")
print("Train:", train["date_parsed"].min(), "to", train["date_parsed"].max())
print("Test: ", test["date_parsed"].min(), "to", test["date_parsed"].max())
print("Unparseable dates (train/test):",
      train["date_parsed"].isna().sum(), "/", test["date_parsed"].isna().sum())

print("\n=== 'Unnamed: 0' COLUMN CHECK ===")
print("Unique values in train:", train["Unnamed: 0"].nunique(), "of", len(train))
print("Min/Max:", train["Unnamed: 0"].min(), train["Unnamed: 0"].max())