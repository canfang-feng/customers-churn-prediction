from pathlib import Path
from PIL import Image
import matplotlib as mpl
import matplotlib.pyplot as plt
import pandas as pd
import streamlit as st


st.set_page_config(
    page_title="Customer Churn Prediction Model Evaluation Dashboard",
    layout="wide",
)

# st.markdown(
#     f"""
#     <style>
#         .css-18e3th9 {{
#             padding-top: {1}rem;
#         }}
#         .css-1gic4dh {{
#             gap: {0}rem;
#         }}
#     </style>
#     """,
#     unsafe_allow_html=True,
# )
st.title("Customer Churn Prediction Model Evaluation Dashboard")
widget_col, display_col = st.columns([1, 9])


def create_experiment_select():
    experiments_path = Path("experiments")
    experiments = sorted([i.stem for i in experiments_path.glob("*")], reverse=False)
    widget = st.sidebar.selectbox("Experiment", experiments)
    return widget


@st.cache
def load_experiment_data(experiment: str):
    experiments_path = Path("experiments")
    output = pd.read_csv(experiments_path / experiment / "test_dataset_outputs.csv")
    stats = pd.read_csv(experiments_path / experiment / "stats.csv")
    try:
        features = pd.read_csv(
            experiments_path / experiment / "feature_importances.csv"
        )
        features = features.sort_values("importances", ascending=False)
        shap_bar = Image.open(experiments_path / experiment / "shap-bar.png")
    except:
        features = None
        shap_bar = None
    confusion_matrix = Image.open(
        experiments_path / experiment / "confusion_matrix.png"
    )
    return output, features, stats, shap_bar, confusion_matrix


st.sidebar.title("Experiment Selection")

with widget_col:
    experiment_select = create_experiment_select()
    output, features, stats, shap_bar, confusion_matrix = load_experiment_data(
        experiment_select
    )

with display_col:
    st.header(f"Model Performance of Experiment {experiment_select}")
    st.dataframe(stats)
    st.image(confusion_matrix)
    if features is not None and shap_bar is not None:
        st.header("Feature Importance of best model from the experiment")
        st.dataframe(features)
        st.image(shap_bar)
