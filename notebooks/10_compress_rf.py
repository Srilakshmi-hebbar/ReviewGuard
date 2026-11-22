import os, sys
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

import joblib
import pandas as pd
from src.features import build_features

PATH = "models/rf_behavioural.joblib"
TMP = "models/rf_compressed_tmp.joblib"

test = pd.read_csv("data/test_clean.csv", sep="\t",
                   parse_dates=["date", "yelpJoinDate"])
X = build_features(test)

model = joblib.load(PATH)
before = model.predict(X)
size_before = os.path.getsize(PATH)

# compress=3 shrinks the file; the model itself is unchanged
joblib.dump(model, TMP, compress=3)
after = joblib.load(TMP).predict(X)
size_after = os.path.getsize(TMP)

same = bool((before == after).all())
print(f"Size before: {size_before / 1e6:.1f} MB")
print(f"Size after:  {size_after / 1e6:.1f} MB")
print("Predictions identical on all test rows:", same)

if same:
    os.replace(TMP, PATH)
    print("Replaced the model with the compressed version.")
else:
    os.remove(TMP)
    print("NOT replaced: predictions differed. Original kept.")