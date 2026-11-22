import io

import numpy as np
import pandas as pd
import streamlit as st
from sklearn.metrics import accuracy_score, confusion_matrix, f1_score

from src.features import build_features
from src.model_utils import MODEL_FILES, load_model

DATE_COLUMNS = ["date", "yelpJoinDate"]
NUMERIC_COLUMNS = [
    "rating", "restaurantRating", "ReviewLength", "friendCount", "reviewCount",
    "firstCount", "usefulCount", "coolCount", "funnyCount", "complimentCount",
    "tipCount", "fanCount",
]
REQUIRED_COLUMNS = DATE_COLUMNS + NUMERIC_COLUMNS
MAX_ROWS = 50000


@st.cache_resource
def get_model(name):
    """Load a saved model once and reuse it across reruns."""
    return load_model(name)


def read_upload(uploaded):
    """Read an uploaded CSV, detecting whether it is tab- or comma-separated."""
    raw = uploaded.getvalue()
    try:
        text = raw.decode("utf-8-sig")
    except UnicodeDecodeError:
        text = raw.decode("latin-1")
    header = text.split("\n", 1)[0]
    sep = "\t" if header.count("\t") > header.count(",") else ","
    return pd.read_csv(io.StringIO(text), sep=sep)


def parse_dates(series):
    try:
        return pd.to_datetime(series, errors="coerce", format="mixed")
    except (ValueError, TypeError):
        return pd.to_datetime(series, errors="coerce")


def prepare(df):
    """Convert columns to the right types and label each row ok / problem."""
    work = pd.DataFrame(index=df.index)
    for col in NUMERIC_COLUMNS:
        work[col] = pd.to_numeric(df[col], errors="coerce")
    work["date"] = parse_dates(df["date"])
    # Training join dates were month-level, so use the first day of the month
    work["yelpJoinDate"] = parse_dates(df["yelpJoinDate"]).dt.to_period("M").dt.to_timestamp()

    status = pd.Series("ok", index=df.index, dtype="object")
    checks = [
        (work[NUMERIC_COLUMNS].isna().any(axis=1), "missing or non-numeric value"),
        (work[DATE_COLUMNS].isna().any(axis=1), "invalid date"),
        ((work[NUMERIC_COLUMNS] < 0).any(axis=1), "negative value"),
        (~work["rating"].between(1, 5), "rating outside 1-5"),
        (work["yelpJoinDate"] > work["date"], "join date after review date"),
    ]
    for mask, message in checks:
        status[mask & (status == "ok")] = message
    return work, status


def page_batch(meta):
    st.title("Batch CSV Analysis")
    st.caption("Upload a file of reviews to score many at once, then download the results.")

    st.info(
        "Results show how closely each review's pattern resembles reviews flagged in the "
        "training data. They do not prove that a review is fake, and the score is not a "
        "calibrated probability."
    )

    with st.expander("Required columns"):
        st.write(", ".join(REQUIRED_COLUMNS))
        st.caption(
            "Dates like 9/17/2010 (review date) and July 2009 (join date) work. The file "
            "can be comma- or tab-separated. Extra columns are kept in the output. If a "
            "'flagged' column (0/1) is present, accuracy is also reported."
        )

    model_name = st.selectbox("Model", list(MODEL_FILES.keys()))
    uploaded = st.file_uploader("Upload a CSV file", type=["csv", "txt", "tsv"])
    if uploaded is None:
        return

    try:
        df = read_upload(uploaded)
    except pd.errors.EmptyDataError:
        st.error("The file is empty.")
        return
    except Exception as e:
        st.error(f"Could not read the file: {e}")
        return

    if len(df) == 0:
        st.error("The file has a header but no rows.")
        return
    if len(df) > MAX_ROWS:
        st.error(f"The file has {len(df):,} rows. The limit is {MAX_ROWS:,}.")
        return

    missing = [c for c in REQUIRED_COLUMNS if c not in df.columns]
    if missing:
        st.error("Missing required columns: " + ", ".join(missing))
        st.caption("Columns found in your file: " + ", ".join(map(str, df.columns)))
        return

    try:
        work, status = prepare(df)
        valid = status == "ok"
        if valid.sum() == 0:
            st.error("No valid rows to analyse. See the reasons below.")
            st.write(status.value_counts())
            return

        X = build_features(work[valid])
        model = get_model(model_name)
        pred = model.predict(X)

        out = df.copy()
        out["row_status"] = status
        out["predicted_flagged"] = pd.Series(pred, index=X.index).astype("Int64")
        out["predicted_label"] = out["predicted_flagged"].map(
            {0: "not flagged", 1: "flagged"})
        if hasattr(model, "predict_proba"):
            scores = model.predict_proba(X)[:, 1]
            out["suspicion_score"] = pd.Series(scores, index=X.index).round(4)
        else:
            margin = model.decision_function(X)
            out["decision_margin"] = pd.Series(margin, index=X.index).round(4)
    except FileNotFoundError as e:
        st.error(str(e))
        return
    except Exception as e:
        st.error(f"Analysis failed: {e}")
        return

    st.subheader("Summary")
    n_flagged = int((pred == 1).sum())
    c1, c2, c3, c4 = st.columns(4)
    c1.metric("Rows in file", f"{len(df):,}")
    c2.metric("Rows analysed", f"{int(valid.sum()):,}")
    c3.metric("Rows skipped", f"{int((~valid).sum()):,}")
    c4.metric("Predicted flagged", f"{n_flagged:,} ({n_flagged / len(pred):.1%})")

    if (~valid).any():
        with st.expander("Why rows were skipped"):
            st.write(status[~valid].value_counts().rename("rows"))

    if "flagged" in df.columns:
        y = pd.to_numeric(df.loc[X.index, "flagged"], errors="coerce")
        keep = y.isin([0, 1])
        if keep.sum() > 0:
            y_true = y[keep].astype(int).values
            y_pred = pd.Series(pred, index=X.index)[keep].values
            st.subheader("Comparison with the 'flagged' column in your file")
            m1, m2 = st.columns(2)
            m1.metric("Accuracy", f"{accuracy_score(y_true, y_pred):.2%}")
            m2.metric("Macro F1", f"{f1_score(y_true, y_pred, average='macro'):.2%}")
            cm = confusion_matrix(y_true, y_pred, labels=[0, 1])
            st.dataframe(
                pd.DataFrame(cm, index=["true 0", "true 1"],
                             columns=["predicted 0", "predicted 1"]))
            st.caption(
                "If this file overlaps the training data, these numbers will look "
                "better than they would on genuinely new data.")

    st.subheader("Results preview")
    st.dataframe(out.head(200), use_container_width=True)
    st.caption("Showing the first 200 rows. The download contains all rows.")

    csv_bytes = out.to_csv(index=False).encode("utf-8")
    st.download_button(
        "Download results as CSV",
        data=csv_bytes,
        file_name="reviewguard_results.csv",
        mime="text/csv",
    )