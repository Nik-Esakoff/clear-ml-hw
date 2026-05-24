import tempfile
from pathlib import Path

import mlflow
import mlflow.sklearn
import numpy as np
from matplotlib import pyplot as plt
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import ConfusionMatrixDisplay

from config import config
from data import get_data
from metrics import evaluate_classifier

def log_confusion_matrix(cm, model) -> None:

    fig, ax = plt.subplots(figsize=(8, 8))

    ConfusionMatrixDisplay(
        confusion_matrix=cm,
        display_labels=model.classes_,
    ).plot(ax=ax)

    ax.set_title("LogisticRegression confusion matrix")

    with tempfile.TemporaryDirectory() as tmp_dir:
        artifact_path = Path(tmp_dir) / "confusion_matrix.png"

        fig.savefig(
            artifact_path,
            bbox_inches="tight",
        )

        mlflow.log_artifact(
            local_path=str(artifact_path),
            artifact_path="plots",
        )

    plt.close(fig)

def log_model_coefficients(model) -> None:

    with tempfile.TemporaryDirectory() as tmp_dir:
        tmp_dir = Path(tmp_dir)

        coefficients_path = tmp_dir / "logistic_regression_coefficients.npy"
        intercept_path = tmp_dir / "logistic_regression_intercept.npy"

        np.save(coefficients_path, model.coef_)
        np.save(intercept_path, model.intercept_)

        mlflow.log_artifact(
            local_path=str(coefficients_path),
            artifact_path="model_artifacts",
        )

        mlflow.log_artifact(
            local_path=str(intercept_path),
            artifact_path="model_artifacts",
        )

def train(model, x_train, y_train) -> None:
    model.fit(x_train, y_train)


def test(model, x_test, y_test) -> None:

    metrics, cm = evaluate_classifier(
        model=model,
        x_test=x_test,
        y_test=y_test,
    )

    mlflow.log_metrics(metrics)

    log_confusion_matrix(
        cm=cm,
        model=model,
    )

    log_model_coefficients(model)

    print("Metrics:", metrics)

if __name__ == "__main__":

    mlflow.set_tracking_uri("http://127.0.0.1:5000")

    mlflow.set_experiment("mlops_4_tracking_digits")

    params = {
        "model": "LogisticRegression",
        "random_state": config["random_state"],
        "max_iter": config["logistic_regression"]["max_iter"],
        "penalty": config["logistic_regression"]["penalty"],
        "C": config["logistic_regression"]["C"],
        "solver": config["logistic_regression"]["solver"],
    }

    with mlflow.start_run(run_name="LogisticRegression_digits"):

        mlflow.log_params(params)

        logistic_regression_model = LogisticRegression(
            max_iter=config["logistic_regression"]["max_iter"],
            penalty=config["logistic_regression"]["penalty"],
            C=config["logistic_regression"]["C"],
            solver=config["logistic_regression"]["solver"],
            random_state=config["random_state"],
        )

        data = get_data()

        train(
            model=logistic_regression_model,
            x_train=data["x_train"],
            y_train=data["y_train"],
        )

        test(
            model=logistic_regression_model,
            x_test=data["x_test"],
            y_test=data["y_test"],
        )

        mlflow.sklearn.log_model(
            sk_model=logistic_regression_model,
            name="model",
        )