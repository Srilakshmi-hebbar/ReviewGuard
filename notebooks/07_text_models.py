import os, sys
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import FunctionTransformer, StandardScaler
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import GroupKFold, cross_val_score
from sklearn.metrics import (accuracy_score, precision_score, recall_score,
                             f1_score, confusion_matrix)
from src.features import build_features, BEHAVIOURAL_FEATURES

dates = ["date", "yelpJoinDate"]
train = pd.read_csv("data/train_clean.csv", sep="\t", parse_dates=dates)
test = pd.read_csv("data/test_clean.csv", sep="\t", parse_dates=dates)


def make_X(df):
    X = build_features(df)
    X["text"] = df["reviewContent"].fillna("").astype(str).values
    return X


X_train, y_train = make_X(train), train["flagged"]
X_test, y_test = make_X(test), test["flagged"]
groups = train["reviewerID"]
unseen = ~test["reviewerID"].isin(set(train["reviewerID"]))


def tfidf():
    return TfidfVectorizer(max_features=5000, ngram_range=(1, 2),
                           min_df=3, sublinear_tf=True)


num_pipe = Pipeline([("log", FunctionTransformer(np.log1p)),
                     ("scale", StandardScaler())])


def lr():
    return LogisticRegression(max_iter=3000, random_state=42)


models = {
    "Text only (TF-IDF + LR)": Pipeline([
        ("prep", ColumnTransformer([("tfidf", tfidf(), "text")])),
        ("clf", lr())]),
    "Combined (TF-IDF + behaviour + LR)": Pipeline([
        ("prep", ColumnTransformer([("tfidf", tfidf(), "text"),
                                    ("num", num_pipe, BEHAVIOURAL_FEATURES)])),
        ("clf", lr())]),
}


def macro(y, p):
    return f1_score(y, p, average="macro")


rows = []
fitted = {}
for name, model in models.items():
    cv = cross_val_score(model, X_train, y_train, groups=groups,
                         cv=GroupKFold(n_splits=5), scoring="f1_macro")
    model.fit(X_train, y_train)
    fitted[name] = model
    pred = model.predict(X_test)
    rows.append({"model": name,
                 "groupCV_macroF1": cv.mean(),
                 "test_acc": accuracy_score(y_test, pred),
                 "test_prec": precision_score(y_test, pred),
                 "test_rec": recall_score(y_test, pred),
                 "test_macroF1": macro(y_test, pred),
                 "unseenReviewers_macroF1": macro(y_test[unseen], pred[unseen])})
    print(f"\n--- {name}: confusion matrix (rows=true 0/1, cols=pred 0/1) ---")
    print(confusion_matrix(y_test, pred))

print("\n=== COMPARISON (behavioural-only LR from Step 8: groupCV 0.8624, "
      "test macroF1 0.8676, unseen 0.8491) ===")
print(pd.DataFrame(rows).set_index("model").round(4).T)

# Which words push toward each class (text-only model)
m = fitted["Text only (TF-IDF + LR)"]
names = m.named_steps["prep"].get_feature_names_out()
coef = m.named_steps["clf"].coef_[0]
order = np.argsort(coef)
print("\n=== Top 15 terms pointing to class 0 ===")
print([names[i].replace("tfidf__", "") for i in order[:15]])
print("\n=== Top 15 terms pointing to class 1 ===")
print([names[i].replace("tfidf__", "") for i in order[::-1][:15]])