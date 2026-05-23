from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score

from config import config
from data import get_data
from metrics import evaluate_classifier
from clearml import Task


def train(model, x_train, y_train) -> None:
    model.fit(x_train, y_train)


def test(model, x_test, y_test, task: Task) -> None:
    metrics, cm = evaluate_classifier(model, x_test, y_test)
    logger = task.get_logger()

    for metric_name, metric_value in metrics.items():
        logger.report_scalar(
            title="metrics",
            series=metric_name,
            value=metric_value,
            iteration=0,
        )

    class_labels = [str(class_id) for class_id in model.classes_]
    logger.report_confusion_matrix(
        title="confusion_matrix",
        series="LogisticRegression",
        matrix=cm,
        iteration=0,
        xaxis="Predicted label",
        yaxis="True label",
        xlabels=class_labels,
        ylabels=class_labels,
    )
    task.upload_artifact(
        name="logistic_regression_coefficients",
        artifact_object=model.coef_,
    )
    task.upload_artifact(
        name="logistic_regression_intercept",
        artifact_object=model.intercept_,
    )

    print("Metrics:", metrics)

if __name__ == "__main__":

    task = Task.init(
        project_name="mlops_4_tracking",
        task_name="LogisticRegression_digits",
        reuse_last_task_id=False,
    )

    params = {
        "model": "LogisticRegression",
        "random_state": config["random_state"],
        "max_iter": config["logistic_regression"]["max_iter"],
        "penalty": config["logistic_regression"]["penalty"],
        "C": config["logistic_regression"]["C"],
        "solver": config["logistic_regression"]["solver"],
    }
    
    task.connect(params)



    logistic_regression_model = LogisticRegression(
        max_iter=config["logistic_regression"]["max_iter"],
        penalty=config["logistic_regression"]["penalty"],
        C=config["logistic_regression"]["C"],
        solver=config["logistic_regression"]["solver"],
        random_state=config["random_state"],
    )

    data = get_data()
    train(logistic_regression_model, data["x_train"], data["y_train"])
    test(logistic_regression_model, data["x_test"], data["y_test"], task,)
