import datetime as dt

import pandas as pd
import streamlit as st

from src.features import build_features
from src.model_utils import MODEL_FILES, load_model


@st.cache_resource
def get_model(name):
    """Load a saved model once and reuse it across reruns."""
    return load_model(name)


def validate(inputs):
    """Return a list of problems with the form values (empty list = valid)."""
    errors = []
    if inputs["join_date"] > inputs["review_date"]:
        errors.append("The join date cannot be after the review date.")
    if inputs["firstCount"] > inputs["reviewCount"]:
        errors.append("'First reviews' cannot be larger than total reviews written.")
    return errors


def page_analyse(meta):
    st.title("Analyse Review")
    st.caption("Enter a review and its reviewer's profile to see what the model predicts.")

    st.info(
        "The models use behavioural features only (no review text). The result shows "
        "how closely the pattern resembles reviews flagged in the training data. "
        "It does not prove a review is fake."
    )

    model_name = st.selectbox("Model", list(MODEL_FILES.keys()))

    with st.form("analyse_form"):
        left, right = st.columns(2)

        with left:
            st.markdown("**The review**")
            rating = st.slider("Star rating given", 1, 5, 5)
            restaurant_rating = st.number_input(
                "Restaurant average rating", min_value=1.0, max_value=5.0,
                value=4.0, step=0.5)
            review_length = st.number_input(
                "Review length (words)", min_value=0, max_value=5000, value=50,
                help="Training lengths were counted AFTER removing common words, "
                     "so a raw review usually looks longer than what the model saw.")
            review_date = st.date_input(
                "Review date", value=dt.date(2012, 1, 1),
                min_value=dt.date(2004, 1, 1), max_value=dt.date.today())
            join_date = st.date_input(
                "Reviewer joined Yelp", value=dt.date(2011, 6, 1),
                min_value=dt.date(2004, 1, 1), max_value=dt.date.today())

        with right:
            st.markdown("**The reviewer's profile**")
            review_count = st.number_input("Total reviews written", 0, 100000, 10)
            first_count = st.number_input("First reviews (first to review a place)", 0, 100000, 1)
            friend_count = st.number_input("Friends", 0, 100000, 5)
            fan_count = st.number_input("Fans", 0, 100000, 0)
            useful_count = st.number_input("Useful votes received", 0, 1000000, 10)
            cool_count = st.number_input("Cool votes received", 0, 1000000, 5)
            funny_count = st.number_input("Funny votes received", 0, 1000000, 3)
            compliment_count = st.number_input("Compliments received", 0, 1000000, 2)
            tip_count = st.number_input("Tips written", 0, 100000, 0)

        submitted = st.form_submit_button("Analyse review")

    if not submitted:
        return

    inputs = {
        "review_date": review_date, "join_date": join_date,
        "reviewCount": int(review_count), "firstCount": int(first_count),
    }
    errors = validate(inputs)
    if errors:
        for e in errors:
            st.error(e)
        return

    if review_date.year < 2005 or review_date.year > 2012:
        st.warning("The training data covers 2005-2012. Dates outside that range "
                   "are outside what the model has seen.")

    try:
        # Join dates in the training data were month-level, so use the 1st of the month
        row = pd.DataFrame([{
            "date": pd.Timestamp(review_date),
            "yelpJoinDate": pd.Timestamp(join_date.replace(day=1)),
            "rating": rating,
            "restaurantRating": restaurant_rating,
            "ReviewLength": int(review_length),
            "friendCount": int(friend_count),
            "reviewCount": int(review_count),
            "firstCount": int(first_count),
            "usefulCount": int(useful_count),
            "coolCount": int(cool_count),
            "funnyCount": int(funny_count),
            "complimentCount": int(compliment_count),
            "tipCount": int(tip_count),
            "fanCount": int(fan_count),
        }])
        X = build_features(row)
        model = get_model(model_name)
        pred = int(model.predict(X)[0])

        if hasattr(model, "predict_proba"):
            score = float(model.predict_proba(X)[0][1])
            score_label = "Suspicion score (model output, not a calibrated probability)"
        else:
            score = None
            margin = float(model.decision_function(X)[0])
            score_label = "Decision margin (positive = leans toward flagged)"
    except FileNotFoundError as e:
        st.error(str(e))
        return
    except Exception as e:
        st.error(f"Prediction failed: {e}")
        return

    st.subheader("Result")
    if pred == 1:
        st.error("This pattern resembles reviews that were FLAGGED (class 1, treated as suspicious).")
    else:
        st.success("This pattern resembles reviews that were NOT flagged (class 0, treated as genuine).")

    if score is not None:
        st.write(score_label)
        st.progress(min(max(score, 0.0), 1.0))
        st.metric("Score for class 1", f"{score * 100:.1f}%")
    else:
        st.metric(score_label, f"{margin:+.2f}")

    with st.expander("Features the model received"):
        st.dataframe(X.T.rename(columns={0: "value"}))