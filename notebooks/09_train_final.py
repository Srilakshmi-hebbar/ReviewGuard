import os, sys, json
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

import numpy as np
import pandas as pd
import joblib
import sklearn
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import FunctionTransformer, StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.svm import LinearSVC
from sklearn.ensemble import RandomForestClassifier
from sklearn.inspection import permutation_importance
from sklearn.model_selection import GroupKFold, cross_val_score
from sklearn.metrics import (accuracy_score, precision_score, recall_score,
                             f1_score, confusion_matrix)
from src.features import build_features, BEHAVIOURAL_FEATURES

os.makedirs("models", exist_ok=True)

dates = ["date", "yelpJoinDate"]
train = pd.read_csv("data/train_clean.csv", sep="\t", parse_dates=dates)
test = pd.read_csv("data/test_clean.csv", sep="\t", parse_dates=dates)

X_train, y_train = build_features(train), train["flagged"]
X_test, y_test = build_features(test), test["flagged"]
groups = train["reviewerID"]
unseen = (~test["reviewerID"].isin(set(train["reviewerID"]))).values


def linear_model(clf):
    return make_pipeline(FunctionTransformer(np.log1p), StandardScaler(), clf)


models = {
    "Logistic Regression": linear_model(LogisticRegression(max_iter=2000, random_state=42)),
    "Linear SVM": linear_model(LinearSVC(max_iter=5000, random_state=42)),
    "Random Forest": RandomForestClassifier(n_estimators=300, random_state=42, n_jobs=-1),
}
files = {
    "Logistic Regression": "logreg_behavioural.joblib",
    "Linear SVM": "svm_behavioural.joblib",
    "Random Forest": "rf_behavioural.joblib",
}


def evaluate(model, X, y, mask):
    pred = model.predict(X)
    y_arr = y.values
    return {
        "accuracy": float(accuracy_score(y_arr, pred)),
        "precision_class1": float(precision_score(y_arr, pred)),
        "recall_class1": float(recall_score(y_arr, pred)),
        "macro_f1": float(f1_score(y_arr, pred, average="macro")),
        "unseen_reviewers_macro_f1": float(
            f1_score(y_arr[mask], pred[mask], average="macro")),
        "confusion_matrix": confusion_matrix(y_arr, pred).tolist(),
    }


# ---- 1) Train, evaluate, save ----
metrics = {}
for name, model in models.items():
    cv = cross_val_score(model, X_train, y_train, groups=groups,
                         cv=GroupKFold(n_splits=5), scoring="f1_macro")
    model.fit(X_train, y_train)
    m = evaluate(model, X_test, y_test, unseen)
    m["group_cv_macro_f1"] = float(cv.mean())
    metrics[name] = m
    joblib.dump(model, os.path.join("models", files[name]))

summary = pd.DataFrame({
    n: {k: v for k, v in m.items() if k != "confusion_matrix"}
    for n, m in metrics.items()
})
print("=== FINAL MODELS (features exclude reviewUsefulCount) ===")
print(summary.round(4))
for n, m in metrics.items():
    print(f"\n{n} confusion matrix (rows=true 0/1, cols=pred 0/1):")
    print(np.array(m["confusion_matrix"]))

# ---- 2) Feature importance (Random Forest) ----
rf = models["Random Forest"]
perm = permutation_importance(rf, X_test, y_test, n_repeats=10,
                              random_state=42, scoring="f1_macro", n_jobs=-1)
imp = pd.DataFrame({
    "feature": BEHAVIOURAL_FEATURES,
    "impurity_importance": rf.feature_importances_,
    "permutation_mean": perm.importances_mean,
    "permutation_std": perm.importances_std,
}).sort_values("permutation_mean", ascending=False)
imp.to_csv("models/feature_importance.csv", index=False)
print("\n=== FEATURE IMPORTANCE (Random Forest) ===")
print(imp.round(4).to_string(index=False))

# ---- 3) Reload check: does the saved file behave identically? ----
loaded = joblib.load(os.path.join("models", files["Random Forest"]))
same = bool((loaded.predict(X_test) == rf.predict(X_test)).all())
print("\nReload check - saved RF gives identical predictions:", same)

# ---- 4) Restaurant-fingerprint test ----
drop = ["restaurantRating", "rating_gap"]
rf2 = RandomForestClassifier(n_estimators=300, random_state=42, n_jobs=-1)
Xtr2, Xte2 = X_train.drop(columns=drop), X_test.drop(columns=drop)
cv2 = cross_val_score(rf2, Xtr2, y_train, groups=groups,
                      cv=GroupKFold(n_splits=5), scoring="f1_macro")
rf2.fit(Xtr2, y_train)
p2 = rf2.predict(Xte2)
print("\n=== RF WITHOUT restaurantRating and rating_gap ===")
print("groupCV macroF1:", round(float(cv2.mean()), 4),
      "(with them:", round(metrics["Random Forest"]["group_cv_macro_f1"], 4), ")")
print("test macroF1:   ", round(float(f1_score(y_test, p2, average="macro")), 4),
      "(with them:", round(metrics["Random Forest"]["macro_f1"], 4), ")")
print("unseen macroF1: ",
      round(float(f1_score(y_test.values[unseen], p2[unseen], average="macro")), 4),
      "(with them:", round(metrics["Random Forest"]["unseen_reviewers_macro_f1"], 4), ")")

# ---- 5) Metadata for the dashboard ----
meta = {
    "features": BEHAVIOURAL_FEATURES,
    "excluded_on_purpose": ["reviewUsefulCount", "IDs", "name", "location",
                            "raw dates", "reviewContent"],
    "label_note": ("flagged=1 is PROVISIONALLY treated as suspicious. This is "
                   "inferred from behaviour, not confirmed by dataset documentation. "
                   "Labels are proxies."),
    "sklearn_version": sklearn.__version__,
    "train_rows": int(len(X_train)),
    "test_rows": int(len(X_test)),
    "metrics": metrics,
}
with open("models/model_metadata.json", "w") as f:
    json.dump(meta, f, indent=2)
print("\nSaved models and metadata in the models folder.")