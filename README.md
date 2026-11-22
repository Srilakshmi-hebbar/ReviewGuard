# ReviewGuard

**Intelligent Fake Review Pattern Detection Using Machine Learning and Behavioural Analytics**

ReviewGuard analyses reviewer behaviour (profile activity, account age, rating patterns) to flag review patterns that resemble reviews marked as suspicious in a Yelp-derived dataset. It includes a Streamlit dashboard with single-review analysis, batch CSV scoring and model analytics.

> **Important:** the labels are proxies, not confirmed fraud. Results below do not prove real-world fraud detection accuracy. See [Limitations](#limitations).

## Features

- **Overview:** headline metrics and known limitations
- **Analyse Review:** score one review from the reviewer's profile, with input validation
- **Model Analytics:** model comparison, confusion matrices, feature importance
- **Batch CSV Analysis:** upload a file, get predictions, download results
- **About Project:** method, results and limitations

## Results

Features exclude `reviewUsefulCount` (see below). Test set: 2,483 reviews.

| Model | Test accuracy | Test macro F1 | Reviewer-grouped CV F1 | Unseen-reviewer F1 |
|---|---|---|---|---|
| Logistic Regression | 86.79% | 86.76% | 86.24% | 84.91% |
| Linear SVM | 86.83% | 86.79% | 86.30% | 84.82% |
| Random Forest | 87.31% | 87.29% | 85.86% | 84.23% |

- *Reviewer-grouped CV* keeps all reviews of one reviewer in the same fold.
- *Unseen-reviewer F1* scores only test rows from reviewers who never appear in training.
- TF-IDF text features were tested: text alone reached about 68% accuracy and added almost nothing to the behavioural features, so the deployed models are behavioural only.
- Including `reviewUsefulCount` raised Random Forest test accuracy to 89.57% in our check. It was excluded because flagged reviews in this data never received useful votes (0% vs about 42% for unflagged), which suggests platform filtering rather than reviewer behaviour.

## Method

1. Clean data, parse dates, check duplicates and missing values
2. Engineer 8 behavioural features (account age at review, reviewed within 30 days of joining, extreme rating, rating gap, per-review ratios, no social ties) on top of 12 raw ones
3. Validate with reviewer-grouped 5-fold cross-validation
4. Train Logistic Regression, Linear SVM (log transform + scaling) and Random Forest
5. Save models with Joblib; the app reuses the same feature code as training

## Limitations

- **Proxy labels:** `flagged` appears to come from Yelp's filtering. Treating class 1 as "suspicious" is inferred from behaviour, not confirmed by documentation.
- **Balanced data:** about 50/50, unlike real platforms.
- **Temporal leakage risk:** profile totals were collected after the reviews were written.
- **Random split:** train and test cover the same dates (2005-2012) and restaurants, so results say nothing about future periods or new restaurants.
- **Reviewer overlap:** 553 reviewers appear in both train and test.
- **Possible filter learning:** behavioural features may partly reflect what the platform's filter looks at.

## Project structure

```
ReviewGuard/
  app/               Streamlit app (app.py and one file per page)
  src/               features.py (feature building), model_utils.py (loading)
  notebooks/         numbered scripts: load, verify, clean, EDA, features, models, training
  models/            saved models, metrics and feature importance
  reports/eda/       exploratory charts
  .streamlit/        dark theme
```

## Setup (Windows Command Prompt)

```
git clone https://github.com/Srilakshmi-hebbar/ReviewGuard.git
cd ReviewGuard
python -m venv venv
venv\Scripts\activate
pip install -r requirements.txt
python -m streamlit run app\app.py
```

## Dataset

The Yelp-derived dataset (`new_data_train.csv`, `new_data_test.csv`, tab-separated, target `flagged`) is **not included** because its licensing is unclear. To retrain, place the files in a `data/` folder and run the numbered scripts in `notebooks/` in order (`03_clean_data.py` creates the cleaned files used by later steps). Pre-trained models are included so the dashboard runs without the data.

## Tech stack

Python, Pandas, NumPy, Scikit-learn, Streamlit, Joblib, Matplotlib.
