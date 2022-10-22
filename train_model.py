from config import train_data_path, test_data_path, label_col, customer_id
from functools import wraps
from pathlib import Path
from time import perf_counter
from typing import Callable, Tuple, Union
from sklearn.svm import LinearSVC
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
import shap
from joblib import dump
from sklearn.ensemble import RandomForestClassifier, HistGradientBoostingClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import (
    classification_report,
    confusion_matrix,
    fbeta_score,
    make_scorer,
)
from sklearn.model_selection import (
    BaseCrossValidator,
    GridSearchCV,
    TimeSeriesSplit,
)
from sklearn.pipeline import Pipeline


def timeit(func):
    @wraps(func)
    def timeit_wrapper(*args, **kwargs):
        start = perf_counter()
        result = func(*args, **kwargs)
        total = perf_counter() - start
        print(f"{func.__name__} took {total:.3f} s")
        return result

    return timeit_wrapper


def get_experiment_id() -> str:
    now = pd.Timestamp.now()
    return f"{now:%Y%m%d%H%M}"


def create_pipeline(model) -> Tuple[Pipeline, dict]:
    pipeline = Pipeline([model.get("classifier")])
    return pipeline, model.get("params")


@timeit
def fit_model(
    pipeline: Pipeline,
    X: np.array,
    y: np.array,
    param_grid: dict,
    scoring: Union[str, Callable] = "f1",
    cv_folds: int = 3,
) -> BaseCrossValidator:
    # cv_splitter = TimeSeriesSplit(cv_folds)

    gs = GridSearchCV(
        pipeline,
        param_grid,
        scoring=scoring,
        cv=cv_folds,
        n_jobs=-1,
        verbose=3,
    )
    gs = gs.fit(X, y)
    return gs


@timeit
def predict_with_sklearn(model: Pipeline, X: np.array) -> np.array:
    return model.predict(X)


def plot_confusion_matrix(cm: np.array, path: Path):
    fig, ax = plt.subplots(1, figsize=(6, 6))
    sns.heatmap(cm, annot=True, ax=ax, fmt="d", cmap="Blues")
    ax.set_title("Confusion matrix")
    ax.set_ylabel("True label")
    ax.set_xlabel("Predicted label")
    fig.tight_layout()
    fig.savefig(path / "confusion_matrix.png")
    plt.close(fig)


def plot_shapley_values(
    model: Pipeline, feature_cols: list[str], X_test: np.array, experiment_path: Path
):
    explainer = shap.TreeExplainer(model.named_steps["clf"], feature_names=feature_cols)
    shap_values = explainer(X_test)
    shap_values.values = shap_values.values[:, :, 1]
    shap_values.base_values = shap_values.base_values[:, 1]
    shap.plots.bar(shap_values, max_display=12, show=False)
    plt.tight_layout()
    plt.savefig(experiment_path / "shap-bar.png")
    plt.close()


def save_outputs(
    model: Pipeline,
    X_test: pd.DataFrame,
    y_pred: pd.Series,
    experiment_path: Path,
):
    outputs = X_test.reset_index()[[customer_id]]
    outputs["pred_label"] = y_pred
    try:
        y_proba = model.predict_proba(X_test)[:, 1]
        outputs["pred_proba"] = y_proba
    except AttributeError:
        pass
    outputs = outputs.dropna(subset=["pred_label"])
    outputs.to_csv(experiment_path / "test_dataset_outputs.csv", index=False)


def main():
    experiment_id = get_experiment_id()
    print(experiment_id)
    experiment_path = Path(f"experiments/{experiment_id}")
    experiment_path.mkdir(parents=True, exist_ok=True)

    print("Loading processed train and test data...")
    train_data = pd.read_csv(train_data_path, index_col=customer_id)
    test_data = pd.read_csv(test_data_path, index_col=customer_id)
    X_train = train_data.drop(columns=label_col)
    y_train = train_data[label_col]
    X_test = test_data.drop(columns=label_col)
    y_test = test_data[label_col]

    feature_cols = X_train.columns

    print("Evaluation metric is: F2 score")
    # The stakes of misidentifying a False Negative (a player was predicted
    # would not churn but then actually did) were much more serious than
    # predicting False Positives. So, give more weight to recall.
    scorer = make_scorer(fbeta_score, beta=2)

    models = {
        "rf": {
            "classifier": ("clf", RandomForestClassifier()),
            "params": {
                "clf__max_depth": [5],
            },
        },
        "gb": {
            "classifier": ("clf", HistGradientBoostingClassifier(scoring=scorer)),
            "params": {
                "clf__max_depth": [5],
            },
        },
        "knc": {
            "classifier": ("clf", KNeighborsClassifier()),
            "params": {
                "clf__weights": ["uniform"],
            },
        },
    }

    print("Training models...")
    stats = []
    best_score = 0
    for model_name, model in models.items():
        pipeline, params = create_pipeline(model)
        gs = fit_model(pipeline, X_train, y_train, params, scoring=scorer)
        if gs.best_score_ > best_score:
            best_score = gs.best_score_
            best_model = gs.best_estimator_
        cv_res = pd.DataFrame(gs.cv_results_)
        cv_res["model_name"] = model_name
        stats.append(cv_res)
    print(f"Best model: {best_model}, score: {best_score}")

    print(f"saving grid search results to {experiment_path}")
    stats = pd.concat(stats, sort=False)
    stats.sort_values("mean_test_score", ascending=False).to_csv(
        experiment_path / "stats.csv", index=False
    )
    print(f"saving best model to {experiment_path} as model.joblib")
    dump(best_model, experiment_path / "model.joblib")

    train_pred = best_model.predict(X_train)
    print(classification_report(train_pred, y_train))
    print(confusion_matrix(y_train, train_pred))
    y_pred = predict_with_sklearn(best_model, X_test)
    cm = confusion_matrix(y_test, y_pred)
    plot_confusion_matrix(cm, experiment_path)
    print(classification_report(y_test, y_pred))
    print(cm)
    save_outputs(best_model, X_test, y_pred, experiment_path)

    try:
        importances = best_model.named_steps["clf"].feature_importances_
        df_importances = pd.DataFrame(
            {"features": feature_cols, "importances": importances}
        )
        df_importances.to_csv(experiment_path / "feature_importances.csv", index=False)
        plot_shapley_values(best_model, feature_cols, X_test, experiment_path)
        print(f"Saved feature importances to {experiment_path}")
    except:
        pass

    try:
        coefs = best_model.named_steps["clf"].coef_
        df_coefs = pd.DataFrame({"features": feature_cols, "coefs": coefs[0]})
        df_coefs.to_csv(experiment_path / "feature_coefs.csv", index=False)
        print(f"Saved feature coefs to {experiment_path}")
    except:
        pass


if __name__ == "__main__":
    main()
