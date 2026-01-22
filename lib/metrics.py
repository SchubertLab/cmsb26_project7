from sklearn.metrics import (
    accuracy_score, precision_score, recall_score,
    roc_auc_score, matthews_corrcoef, balanced_accuracy_score, average_precision_score
)


def calc_metrics(y_true, y_prob):
    y_pred = (y_prob >= 0.5).astype(int)

    metrics = {
        "ACC": accuracy_score(y_true, y_pred),
        "Balanced_ACC": balanced_accuracy_score(y_true, y_pred),
        "Precision": precision_score(y_true, y_pred),
        "Recall": recall_score(y_true, y_pred),
        "AUROC": roc_auc_score(y_true, y_prob),
        "AUPRC": average_precision_score(y_true, y_prob),
        "MCC": matthews_corrcoef(y_true, y_pred)
    }
    return metrics








