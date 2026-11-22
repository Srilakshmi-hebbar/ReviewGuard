import streamlit as st


def page_about(meta):
    st.title("About Project")
    st.caption("ReviewGuard: Intelligent Fake Review Pattern Detection Using "
               "Machine Learning and Behavioural Analytics")

    st.subheader("What it does")
    st.markdown(
        "ReviewGuard analyses reviewer behaviour (profile activity, account age, rating "
        "patterns) to find review patterns that resemble reviews flagged in a "
        "Yelp-derived dataset. It offers single-review analysis, batch CSV scoring "
        "with downloadable results, and model analytics."
    )

    st.subheader("Data")
    st.markdown(
        f"- Yelp-derived dataset, tab-separated, target column `flagged`\n"
        f"- {meta['train_rows']:,} training reviews and {meta['test_rows']:,} test reviews\n"
        "- Classes are roughly balanced (about 50/50)\n"
        "- Cleaning: dropped a leftover index column, filled 3 missing review texts and "
        "1 missing location, parsed dates, checked for duplicates (none found)"
    )

    st.subheader("Method")
    st.markdown(
        "1. **Features:** 12 raw behavioural columns plus 8 engineered ones (account age "
        "at review, reviewed within 30 days of joining, extreme rating, rating gap, "
        "ratios per review, no social ties).\n"
        "2. **Excluded on purpose:** `reviewUsefulCount` (it reflects platform filtering), "
        "IDs, names, locations, raw dates and review text.\n"
        "3. **Validation:** reviewer-grouped 5-fold cross-validation, so no reviewer "
        "appears in both the training and validation part of a fold.\n"
        "4. **Models:** Logistic Regression and Linear SVM (log transform + scaling) and "
        "a Random Forest.\n"
        "5. **Text:** TF-IDF was tested. Text alone reached about 68% accuracy and added "
        "almost nothing to the behavioural features, so the deployed models are "
        "behavioural only.\n"
        "6. **Persistence:** trained models are saved with Joblib; the app reuses the "
        "same feature-building code as training."
    )

    st.subheader("Results (test set)")
    rows = {}
    for name, m in meta["metrics"].items():
        rows[name] = {
            "Test accuracy": f"{m['accuracy']:.2%}",
            "Test macro F1": f"{m['macro_f1']:.2%}",
            "Reviewer-grouped CV F1": f"{m['group_cv_macro_f1']:.2%}",
            "Unseen-reviewer F1": f"{m['unseen_reviewers_macro_f1']:.2%}",
        }
    st.table(rows)
    st.caption(
        "Including `reviewUsefulCount` raised the Random Forest to 89.57% test accuracy "
        "in our check, which is why the earlier 89% figure is not used as the headline."
    )

    st.subheader("Limitations")
    st.warning(
        "These results do not prove real-world fraud detection accuracy."
    )
    st.markdown(
        "- **Proxy labels:** `flagged` appears to come from Yelp's filtering, not "
        "confirmed fraud. Class 1 is treated as 'suspicious' based on behavioural "
        "evidence (inferred, not documented).\n"
        "- **Balanced data:** real platforms have far fewer fake reviews.\n"
        "- **Temporal leakage risk:** profile totals were collected after the reviews "
        "were written, so a review carries information from its reviewer's future.\n"
        "- **Random, not temporal, split:** train and test cover the same dates "
        "(2005-2012), so results say nothing about future reviews.\n"
        "- **Same restaurants:** every test row comes from a restaurant that also "
        "appears in training.\n"
        "- **Reviewer overlap:** 553 reviewers appear in both train and test, which is "
        "why we also report unseen-reviewer scores.\n"
        "- **Possible filter learning:** behavioural features may partly reflect "
        "what the platform's filter looks at.\n"
        "- **Text length:** `ReviewLength` was counted on stopword-removed text."
    )

    st.subheader("Technology")
    st.markdown("Python, Pandas, NumPy, Scikit-learn, Streamlit, Joblib, Matplotlib.")