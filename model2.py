from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import accuracy_score

from config import config
from data import get_data
from metrics import evaluate_classifier
from clearml import Task

def train(model, x_train, y_train) -> None:
    model.fit(x_train, y_train)


def test(model, x_test, y_test, task: Task) -> None:
    metrics, cm = evaluate_classifier(model, x_test, y_test)
    metrics["actual_tree_depth"] = model.get_depth()

    metrics["actual_number_of_leaves"] = model.get_n_leaves()    
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
        series="DecisionTreeClassifier",
        matrix=cm,
        iteration=0,
        xaxis="Predicted label",
        yaxis="True label",
        xlabels=class_labels,
        ylabels=class_labels,
    )

    print("Metrics:", metrics)


if __name__ == "__main__":

    task = Task.init(
        project_name="mlops_4_tracking",
        task_name="DecisionTree_digits",
        reuse_last_task_id=False,
    )
    params = {
        "model": "DecisionTreeClassifier",
        "random_state": config["random_state"],
        "max_depth": config["decision_tree"]["max_depth"],
        "criterion": config["decision_tree"]["criterion"],
    }

    task.connect(params)



    decision_tree_model = DecisionTreeClassifier(
        random_state=config["random_state"],
        max_depth=config["decision_tree"]["max_depth"],
        criterion=config["decision_tree"]["criterion"],
    )

    data = get_data()
    train(decision_tree_model, data["x_train"], data["y_train"])
    test(decision_tree_model, data["x_test"], data["y_test"], task)
