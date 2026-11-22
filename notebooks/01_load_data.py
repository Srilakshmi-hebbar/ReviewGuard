import pandas as pd

# Make pandas print wider tables so we can see all columns
pd.set_option("display.max_columns", None)
pd.set_option("display.width", 200)
pd.set_option("display.max_colwidth", 60)

TRAIN_PATH = "data/new_data_train.csv"
TEST_PATH = "data/new_data_test.csv"

# The files are tab-separated, so we must pass sep="\t"
train = pd.read_csv(TRAIN_PATH, sep="\t")
test = pd.read_csv(TEST_PATH, sep="\t")

print("=== SHAPES (rows, columns) ===")
print("Train:", train.shape)
print("Test: ", test.shape)

print("\n=== COLUMNS AND DATA TYPES (train) ===")
print(train.dtypes)

print("\n=== SAME COLUMNS IN TRAIN AND TEST? ===")
print(list(train.columns) == list(test.columns))

print("\n=== FIRST 3 ROWS (train) ===")
print(train.head(3))

print("\n=== TARGET: flagged (train) ===")
print(train["flagged"].value_counts(dropna=False))

print("\n=== TARGET: flagged (test) ===")
print(test["flagged"].value_counts(dropna=False))

print("\n=== MISSING VALUES PER COLUMN (train) ===")
print(train.isna().sum())

print("\n=== DUPLICATE ROWS (train) ===")
print(train.duplicated().sum())