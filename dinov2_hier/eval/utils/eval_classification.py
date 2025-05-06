from sklearn.metrics import (accuracy_score, balanced_accuracy_score,
                             classification_report, f1_score, log_loss)
def eval_classification(test_labels, test_predictions):

    accuracy = accuracy_score(test_labels, test_predictions)
    balanced_acc = balanced_accuracy_score(test_labels, test_predictions)
    weighted_f1 = f1_score(test_labels, test_predictions, average="weighted")
    return accuracy, balanced_acc, weighted_f1