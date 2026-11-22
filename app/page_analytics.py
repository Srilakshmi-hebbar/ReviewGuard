import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import streamlit as st

from src.model_utils import load_importance

BG, PANEL = "#0b1220", "#131c2e"
CYAN, RED, TEXT = "#00c2ff", "#ff4d6d", "#e6edf3"


def _style(fig, ax):
    """Make a matplotlib chart match the dark dashboard theme."""
    fig.patch.set_facecolor(BG)
    ax.set_facecolor(PANEL)
    ax.tick_params(colors=TEXT)
    for spine in ax.spines.values():
        spine.set_color("#22304a")
    ax.xaxis.label.set_color(TEXT)
    ax.yaxis.label.set_color(TEXT)
    ax.title.set_color(TEXT)


def _comparison_table(meta):
    rows = {}
    for name, m in meta["metrics"].items():
        rows[name] = {
            "Test accuracy": m["accuracy"],
            "Precision (class 1)": m["precision_class1"],
            "Recall (class 1)": m["recall_class1"],
            "Test macro F1": m["macro_f1"],
            "Reviewer-grouped CV F1": m["group_cv_macro_f1"],
            "Unseen-reviewer F1": m["unseen_reviewers_macro_f1"],
        }
    return pd.DataFrame(rows).T


def _confusion_matrix_chart(cm):
    fig, ax = plt.subplots(figsize=(5, 4))
    ax.imshow(cm, cmap="Blues")
    labels = ["0 (not flagged)", "1 (flagged)"]
    ax.set_xticks([0, 1])
    ax.set_yticks([0, 1])
    ax.set_xticklabels(labels)
    ax.set_yticklabels(labels)
    ax.set_xlabel("Predicted")
    ax.set_ylabel("True")
    ax.set_title("Confusion matrix (test set)")
    top = cm.max()
    for i in range(2):
        for j in range(2):
            colour = "white" if cm[i, j] > top / 2 else "black"
            ax.text(j, i, f"{cm[i, j]:,}", ha="center", va="center",
                    color=colour, fontsize=14, fontweight="bold")
    _style(fig, ax)
    fig.tight_layout()
    return fig


def _importance_chart(df, column, top_n):
    d = df.sort_values(column, ascending=False).head(top_n)
    d = d.iloc[::-1]  # reverse so the biggest bar is at the top
    colours = [CYAN if v >= 0 else RED for v in d[column]]
    fig, ax = plt.subplots(figsize=(7, 0.4 * top_n + 1.5))
    ax.barh(d["feature"], d[column], color=colours)
    ax.set_xlabel(column.replace("_", " "))
    ax.set_title("Random Forest feature importance")
    _style(fig, ax)
    fig.tight_layout()
    return fig


def page_analytics(meta):
    st.title("Model Analytics")
    st.caption("How the three models performed, and what the Random Forest relies on.")

    st.subheader("Model comparison")
    table = _comparison_table(meta)
    st.dataframe(table.style.format("{:.2%}"), use_container_width=True)
    st.caption(
        "The three models are within about one point of each other. 'Reviewer-grouped "
        "CV' keeps all reviews of one reviewer in the same fold; 'Unseen-reviewer' scores "
        "only test rows from reviewers that never appear in training. These are stricter "
        "than the plain test score."
    )

    st.subheader("Confusion matrix")
    model_name = st.selectbox("Model", list(meta["metrics"].keys()), key="cm_model")
    cm = np.array(meta["metrics"][model_name]["confusion_matrix"])
    tn, fp, fn, tp = int(cm[0][0]), int(cm[0][1]), int(cm[1][0]), int(cm[1][1])

    left, right = st.columns([1, 1])
    with left:
        fig = _confusion_matrix_chart(cm)
        st.pyplot(fig)
        plt.close(fig)
    with right:
        st.markdown("**Reading it** (rows = true class, columns = predicted)")
        st.metric("Class 1 caught (recall)", f"{tp / (tp + fn):.1%}")
        st.metric("Class 0 correctly cleared (recall)", f"{tn / (tn + fp):.1%}")
        st.metric("Genuine reviews wrongly flagged", f"{fp:,}")
        st.metric("Flagged reviews missed", f"{fn:,}")

    st.subheader("Feature importance (Random Forest)")
    try:
        imp = load_importance()
    except FileNotFoundError as e:
        st.error(str(e))
        return

    c1, c2 = st.columns(2)
    measure = c1.radio(
        "Measure",
        ["permutation_mean", "impurity_importance"],
        format_func=lambda x: "Permutation (drop in macro F1 when shuffled)"
        if x == "permutation_mean" else "Impurity (split-based)",
    )
    top_n = c2.slider("Features to show", 5, len(imp), 12)

    fig = _importance_chart(imp, measure, top_n)
    st.pyplot(fig)
    plt.close(fig)

    st.info(
        "How to read this: many profile features are strongly correlated, so when one is "
        "shuffled the others compensate and every individual score looks small. Impurity "
        "importance favours features with many distinct values. Features near zero or "
        "negative are not proven useless. Importance describes what the model uses, not "
        "what makes a review fake."
    )

    with st.expander("Why reviewUsefulCount is excluded"):
        st.markdown(
            "Flagged reviews in this data never received a useful vote (0% vs about 42% "
            "for unflagged), which suggests the platform's filtering, not reviewer "
            "behaviour, produced that pattern. In our test, adding it to the Random "
            "Forest raised test accuracy from 87.31% to 89.57%, and unseen-reviewer "
            "macro F1 from 84.23% to 87.02%. We excluded it so the model does not "
            "learn the filter itself."
        )