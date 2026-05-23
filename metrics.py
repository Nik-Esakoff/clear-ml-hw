from sklearn.metrics import (
    accuracy_score,
    confusion_matrix,
    f1_score,
    roc_auc_score,
)


def evaluate_classifier(model, x_test, y_test):
    y_pred = model.predict(x_test)
    y_proba = model.predict_proba(x_test)

    metrics = {
        "accuracy": accuracy_score(y_true=y_test, y_pred=y_pred),
        "f1_macro": f1_score(y_true=y_test, y_pred=y_pred, average="macro"),
        "auc_roc_ovr_macro": roc_auc_score(
            y_true=y_test,
            y_score=y_proba,
            multi_class="ovr",
            average="macro",
        ),
    }

    cm = confusion_matrix(y_true=y_test, y_pred=y_pred)

    return metrics, cm
