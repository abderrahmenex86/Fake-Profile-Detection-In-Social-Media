from sklearn.metrics import (accuracy_score, balanced_accuracy_score,
                             classification_report, cohen_kappa_score,
                             f1_score, log_loss, matthews_corrcoef,
                             precision_score, recall_score, roc_auc_score)


def compute_metrics(y_true, y_pred, y_prob=None, average="macro"):
    metrics = {
        "Accuracy": accuracy_score(y_true, y_pred),
        "Precision": precision_score(y_true, y_pred, average=average, zero_division=0),
        "Recall": recall_score(y_true, y_pred, average=average, zero_division=0),
        "F1": f1_score(y_true, y_pred, average=average, zero_division=0),
        "CohenKappa": cohen_kappa_score(y_true, y_pred),
        "MatthewsCorrCoef": matthews_corrcoef(y_true, y_pred),
        "BalancedAccuracy": balanced_accuracy_score(y_true, y_pred),
    }
    if y_prob is not None:
        try:
            metrics["ROC_AUC"] = compute_roc_auc(y_true, y_prob)
            metrics["LogLoss"] = log_loss(y_true, y_prob)
        except Exception as e:
            metrics["ROC_AUC"] = f"Error: {str(e)}"
            metrics["LogLoss"] = f"Error: {str(e)}"
    return metrics


def print_classification_report(y_true, y_pred, target_names=None):
    print(
        classification_report(
            y_true, y_pred, target_names=target_names, zero_division=0
        )
    )


def compute_roc_auc(y_true, y_prob, multi_class="ovr", average="macro"):
    n_classes = len(set(y_true))

    if n_classes == 2:
        if y_prob.shape[1] == 2:
            return roc_auc_score(y_true, y_prob[:, 1])
        else:
            return roc_auc_score(y_true, y_prob)
    else:
        return roc_auc_score(y_true, y_prob, multi_class=multi_class, average=average)
