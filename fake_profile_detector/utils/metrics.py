from sklearn.metrics import (accuracy_score, balanced_accuracy_score,
                             classification_report, cohen_kappa_score,
                             f1_score, log_loss, matthews_corrcoef,
                             precision_score, recall_score, roc_auc_score)

ROUND = 4


def compute_metrics(y_true, y_pred, y_prob=None, average="binary"):
    metrics = {
        "Accuracy": round(accuracy_score(y_true, y_pred), ROUND),
        "Precision": round(precision_score(y_true, y_pred, average=average), ROUND),
        "Recall": round(recall_score(y_true, y_pred, average=average), ROUND),
        "F1": round(f1_score(y_true, y_pred, average=average), ROUND),
        "CohenKappa": round(cohen_kappa_score(y_true, y_pred).item(), ROUND),
        "MatthewsCorrCoef": round(matthews_corrcoef(y_true, y_pred), ROUND),
        "BalancedAccuracy": round(
            balanced_accuracy_score(y_true, y_pred).item(), ROUND
        ),
    }
    # if y_prob is not None:
    #     # for binary: pass y_prob[:,1]; for multiclass: full matrix
    #     metrics["ROC_AUC"] = round(roc_auc_score(y_true, y_prob[:, 1]), ROUND)
    #     metrics["LogLoss"] = round(log_loss(y_true, y_prob), ROUND)
    return metrics


def print_classification_report(y_true, y_pred, target_names=None):
    print(classification_report(y_true, y_pred, target_names=target_names))


def compute_roc_auc(y_true, y_prob, multi_class="ovr"):
    return roc_auc_score(y_true, y_prob, multi_class=multi_class)
