import pandas as pd

def clean(df):
    df = df.copy()
    # Old row index, not a real feature
    df = df.drop(columns=["Unnamed: 0"])
    # Keep the rows, fill the few gaps (3 texts, 1 location in train)
    df["reviewContent"] = df["reviewContent"].fillna("")
    df["location"] = df["location"].fillna("Unknown")
    # Convert text dates to real dates so we can do arithmetic on them
    df["date"] = pd.to_datetime(df["date"], format="%m/%d/%Y")
    df["yelpJoinDate"] = pd.to_datetime(
        df["yelpJoinDate"], format="%B %Y", errors="coerce"
    )
    return df

train = clean(pd.read_csv("data/new_data_train.csv", sep="\t"))
test = clean(pd.read_csv("data/new_data_test.csv", sep="\t"))

for name, df in [("TRAIN", train), ("TEST", test)]:
    print(f"\n===== {name} {df.shape} =====")
    print("Missing values left:", int(df.isna().sum().sum()))
    print("Unparseable join dates:", int(df["yelpJoinDate"].isna().sum()))
    age_days = (df["date"] - df["yelpJoinDate"]).dt.days
    print("Reviews dated BEFORE the join month:", int((age_days < 0).sum()))
    print("Account age at review (days), describe:")
    print(age_days.describe().round(1))

    words = df["reviewContent"].str.split().str.len()
    print("ReviewLength equals word count of text:",
          round(float((words == df["ReviewLength"]).mean()), 3))

print("\n=== reviewUsefulCount > 0, share by class (train) ===")
print((train["reviewUsefulCount"] > 0).groupby(train["flagged"]).mean().round(4))

train.to_csv("data/train_clean.csv", sep="\t", index=False)
test.to_csv("data/test_clean.csv", sep="\t", index=False)
print("\nSaved data/train_clean.csv and data/test_clean.csv")