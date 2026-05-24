import tempfile
from pathlib import Path

import mlflow
import mlflow.sklearn
from matplotlib import pyplot as plt
from sklearn.metrics import ConfusionMatrixDisplay
from sklearn.tree import DecisionTreeClassifier

from config import config
from data import get_data
from metrics import evaluate_classifier


def train(model, x_train, y_train) -> None:
    model.fit(x_train, y_train)


def log_confusion_matrix(cm, model) -> None:

    fig, ax = plt.subplots(figsize=(8, 8))

    ConfusionMatrixDisplay(
        confusion_matrix=cm,
        display_labels=model.classes_,
    ).plot(ax=ax)

    ax.set_title("DecisionTreeClassifier confusion matrix")

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


def test(model, x_test, y_test) -> None:
    metrics, cm = evaluate_classifier(
        model=model,
        x_test=x_test,
        y_test=y_test,
    )
    
    metrics["actual_tree_depth"] = model.get_depth()
    metrics["actual_number_of_leaves"] = model.get_n_leaves()

    mlflow.log_metrics(metrics)

    log_confusion_matrix(
        cm=cm,
        model=model,
    )

    print("Metrics:", metrics)


if __name__ == "__main__":
    mlflow.set_tracking_uri("http://127.0.0.1:5000")
    mlflow.set_experiment("mlops_4_tracking_digits")

    params = {
        "model": "DecisionTreeClassifier",
        "random_state": config["random_state"],
        "max_depth": config["decision_tree"]["max_depth"],
        "criterion": config["decision_tree"]["criterion"],
    }

    with mlflow.start_run(run_name="DecisionTree_digits"):
        mlflow.log_params(params)

        decision_tree_model = DecisionTreeClassifier(
            random_state=config["random_state"],
            max_depth=config["decision_tree"]["max_depth"],
            criterion=config["decision_tree"]["criterion"],
        )

        data = get_data()

        train(
            model=decision_tree_model,
            x_train=data["x_train"],
            y_train=data["y_train"],
        )

        test(
            model=decision_tree_model,
            x_test=data["x_test"],
            y_test=data["y_test"],
        )

        mlflow.sklearn.log_model(
            sk_model=decision_tree_model,
            name="model",
        )