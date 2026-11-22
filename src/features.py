import pandas as pd

# Raw columns we keep as behavioural features
RAW_BEHAVIOUR_COLS = [
    "rating", "restaurantRating", "ReviewLength",
    "friendCount", "reviewCount", "firstCount", "usefulCount",
    "coolCount", "funnyCount", "complimentCount", "tipCount", "fanCount",
]

# New features we compute from the raw ones
ENGINEERED_COLS = [
    "account_age_days",        # days between joining and writing the review
    "reviewed_within_30d",     # 1 if the review was written within 30 days of joining
    "is_extreme_rating",       # 1 if the rating is 1 or 5 stars
    "rating_gap",              # |reviewer rating - restaurant average rating|
    "friends_per_review",      # social ties relative to activity
    "useful_per_review",       # community approval relative to activity
    "compliments_per_review",  # compliments relative to activity
    "no_social_ties",          # 1 if the reviewer has 0 friends AND 0 fans
]

BEHAVIOURAL_FEATURES = RAW_BEHAVIOUR_COLS + ENGINEERED_COLS


def build_features(df):
    """Turn a cleaned review DataFrame into the behavioural feature table.

    Expects the columns produced by our cleaning step, with 'date' and
    'yelpJoinDate' already converted to real dates.
    """
    out = pd.DataFrame(index=df.index)

    for col in RAW_BEHAVIOUR_COLS:
        out[col] = df[col]

    age = (df["date"] - df["yelpJoinDate"]).dt.days
    out["account_age_days"] = age.clip(lower=0)
    out["reviewed_within_30d"] = (out["account_age_days"] <= 30).astype(int)

    out["is_extreme_rating"] = df["rating"].isin([1, 5]).astype(int)
    out["rating_gap"] = (df["rating"] - df["restaurantRating"]).abs()

    # The +1 avoids dividing by zero for reviewers with 0 reviews
    reviews = df["reviewCount"] + 1
    out["friends_per_review"] = df["friendCount"] / reviews
    out["useful_per_review"] = df["usefulCount"] / reviews
    out["compliments_per_review"] = df["complimentCount"] / reviews

    out["no_social_ties"] = (
        (df["friendCount"] == 0) & (df["fanCount"] == 0)
    ).astype(int)

    return out[BEHAVIOURAL_FEATURES]