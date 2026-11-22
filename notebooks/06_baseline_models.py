import os, sys
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

import numpy as np
import pandas as pd
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import FunctionTransformer, StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.svm import LinearSVC
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import GroupKFold, cross_val_score
from sklearn.metrics import (accuracy_score, precision_score, recall_score,
                             f1_score, confusion_matrix)
from src.features import build_features

dates = ["date", "yelpJoinDate"]
train = pd.read_csv("data/train_clean.csv", sep="\t", parse_dates=dates)
test = pd.read_csv("data/test_clean.csv", sep="\t", parse_dates=dates)

X_train, y_train = build_features(train), train["flagged"]
X_test, y_test = build_features(test), test["flagged"]
groups = train["reviewerID"]

# Test rows whose reviewer never appears in train = truly unseen reviewers
unseen = ~test["reviewerID"].isin(set(train["reviewerID"]))
print("Test rows:", len(test), "| from reviewers unseen in train:", int(unseen.sum()))

# Skewed counts -> log1p, then scale. Trees don't need this.
def linear_model(clf):
    return make_pipeline(FunctionTransformer(np.log1p), StandardScaler(), clf)

models = {
    "Logistic Regression": linear_model(LogisticRegression(max_iter=2000, random_state=42)),
    "Linear SVM": linear_model(LinearSVC(max_iter=5000, random_state=42)),
    "Random Forest": RandomForestClassifier(n_estimators=300, random_state=42, n_jobs=-1),
}

def scores(y_true, y_pred):
    return {
        "acc": accuracy_score(y_true, y_pred),
        "prec": precision_score(y_true, y_pred),
        "rec": recall_score(y_true, y_pred),
        "macroF1": f1_score(y_true, y_pred, average="macro"),
    }

gkf = GroupKFold(n_splits=5)
rows = []
for name, model in models.items():
    cv = cross_val_score(model, X_train, y_train, groups=groups,
                         cv=gkf, scoring="f1_macro")
    model.fit(X_train, y_train)
    pred = model.predict(X_test)
    s_all = scores(y_test, pred)
    s_new = scores(y_test[unseen], pred[unseen])
    rows.append({"model": name,
                 "groupCV_macroF1": cv.mean(),
                 "test_acc": s_all["acc"], "test_prec": s_all["prec"],
                 "test_rec": s_all["rec"], "test_macroF1": s_all["macroF1"],
                 "unseenReviewers_macroF1": s_new["macroF1"]})
    print(f"\n--- {name}: confusion matrix on test (rows=true 0/1, cols=pred 0/1) ---")
    print(confusion_matrix(y_test, pred))

print("\n=== MODEL COMPARISON ===")
print(pd.DataFrame(rows).set_index("model").round(4).T)