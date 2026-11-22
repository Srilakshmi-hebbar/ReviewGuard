import importlib
import sys
from pathlib import Path

# Let the app import from the project's src and app folders
ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(Path(__file__).resolve().parent))

import streamlit as st

from src.model_utils import load_metadata

st.set_page_config(page_title="ReviewGuard", page_icon="🛡️", layout="wide")

st.markdown(
    """
    <style>
    [data-testid="stMetric"] {
        background-color: #131c2e;
        border: 1px solid #22304a;
        border-radius: 10px;
        padding: 14px 16px;
    }
    h1, h2, h3 { letter-spacing: 0.3px; }
    </style>
    """,
    unsafe_allow_html=True,
)


@st.cache_data
def get_metadata():
    return load_metadata()


def page_overview(meta):
    st.title("🛡️ ReviewGuard")
    st.caption("Fake review pattern detection using machine learning and behavioural analytics")

    st.warning(
        "Labels in this dataset are proxies (inferred from Yelp's filtering), and "
        "class 1 is provisionally treated as 'suspicious'. Results below do not "
        "prove real-world fraud detection accuracy."
    )

    rf = meta["metrics"]["Random Forest"]
    st.subheader("Random Forest (behavioural features)")
    c1, c2, c3, c4 = st.columns(4)
    c1.metric("Test accuracy", f"{rf['accuracy'] * 100:.2f}%")
    c2.metric("Test macro F1", f"{rf['macro_f1'] * 100:.2f}%")
    c3.metric("Reviewer-grouped CV F1", f"{rf['group_cv_macro_f1'] * 100:.2f}%")
    c4.metric("Unseen-reviewer F1", f"{rf['unseen_reviewers_macro_f1'] * 100:.2f}%")

    st.subheader("Dataset and features")
    d1, d2, d3 = st.columns(3)
    d1.metric("Training reviews", f"{meta['train_rows']:,}")
    d2.metric("Test reviews", f"{meta['test_rows']:,}")
    d3.metric("Behavioural features", len(meta["features"]))

    with st.expander("Known limitations"):
        st.markdown(
            "- Labels are proxies for suspicious behaviour, not confirmed fraud.\n"
            "- The data is balanced (about 50/50), unlike real platforms.\n"
            "- Profile totals were collected after the reviews were written "
            "(possible temporal leakage).\n"
            "- Train and test cover the same dates and restaurants, so results "
            "say nothing about future periods or new restaurants.\n"
            "- `reviewUsefulCount` was excluded because it reflects platform "
            "filtering, not reviewer behaviour."
        )


def page_placeholder(title):
    st.title(title)
    st.info("This page is built in a later step.")


# Page name -> (module file name, function name). Pages without a file yet
# show the placeholder automatically.
PAGE_MODULES = {
    "Analyse Review": ("page_analyse", "page_analyse"),
    "Model Analytics": ("page_analytics", "page_analytics"),
    "Batch CSV Analysis": ("page_batch", "page_batch"),
    "About Project": ("page_about", "page_about"),
}
PAGES = ["Overview"] + list(PAGE_MODULES.keys())


def run_page(choice, meta):
    if choice == "Overview":
        page_overview(meta)
        return
    module_name, func_name = PAGE_MODULES[choice]
    try:
        module = importlib.import_module(module_name)
    except ModuleNotFoundError as e:
        if e.name == module_name:  # the page file does not exist yet
            page_placeholder(choice)
            return
        raise
    getattr(module, func_name)(meta)


st.sidebar.title("ReviewGuard")
choice = st.sidebar.radio("Navigate", PAGES)

try:
    meta = get_metadata()
except FileNotFoundError as e:
    st.error(str(e))
    st.stop()

run_page(choice, meta)