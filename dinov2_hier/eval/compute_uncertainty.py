import pandas as pd
from sklearn.metrics import (accuracy_score, balanced_accuracy_score,
                             classification_report, f1_score, log_loss)
import numpy as np

import warnings
warnings.filterwarnings("ignore")



def bootstrap_uncertainty(true, pred, n_bootstrap=1000, alpha=0.05):
    """Compute bootstrap confidence intervals for balanced accuracy and weighted F1-score, handling missing classes."""
    true, pred = np.array(true), np.array(pred)
    n = len(true)
    
    unique_classes = np.unique(true)  # Get all classes from the true labels
    bal_acc_scores = []
    weighted_f1_scores = []
    
    for _ in range(n_bootstrap):
        while True:  # Repeat sampling until all classes are included
            indices = np.random.choice(np.arange(n), size=n, replace=True)
            true_sampled, pred_sampled = true[indices], pred[indices]
            
            if set(unique_classes).issubset(set(true_sampled)):  # Ensure all classes exist
                break
        
        # Compute metrics
        bal_acc_scores.append(balanced_accuracy_score(true_sampled, pred_sampled))
        weighted_f1_scores.append(f1_score(true_sampled, pred_sampled, average="weighted", labels=unique_classes))

    
    # Compute confidence intervals
    bal_acc_ci = np.percentile(bal_acc_scores, [100 * alpha / 2, 100 * (1 - alpha / 2)])
    weighted_f1_ci = np.percentile(weighted_f1_scores, [100 * alpha / 2, 100 * (1 - alpha / 2)])

    return {
        "weighted_f1": {"mean": np.mean(weighted_f1_scores), "int": (weighted_f1_ci[1] - weighted_f1_ci[0])/2},
        "balanced_accuracy": {"mean": np.mean(bal_acc_scores), "int": (bal_acc_ci[1] - bal_acc_ci[0])/2 }
    }


# BM
for root in root_list:
    knn_file = root + 'knn_eval/knn_eval_labels_and_predictions.csv'
    log_file = root + 'log_reg_eval/log_reg_eval_labels_and_predictions.csv'
    df_knn = pd.read_csv(knn_file)
    df_log = pd.read_csv(log_file)
    for i,df in enumerate([df_knn, df_log]):
        true, predict = df['True Labels'].tolist(), df['Predicted Labels'].tolist()
        result = bootstrap_uncertainty(true, predict)
        print('knn' if i == 0 else 'log')
        for key in result:
            print(key, result[key]['mean'], result[key]["int"] )



# Acevedo
for root in root_list:

    bacc_log = []
    bacc_knn = []
    wf1_log = []
    wf1_knn = []
    for k in range(5):
        print(k)
        knn_file = root + 'dino_eval_knn_eval/dino_eval_knn_eval_logreg_labels_and_predictions_fold_{}.csv'.format(str(k))
        log_file = root + 'dino_eval_log_reg_eval/dino_eval_log_reg_eval_logreg_labels_and_predictions_fold_{}.csv'.format(str(k))
        df_knn = pd.read_csv(knn_file)
        df_log = pd.read_csv(log_file)
        for i,df in enumerate([df_knn, df_log]):
            true, predict = df['label'].tolist(), df['prediction'].tolist()
            balanced_acc = balanced_accuracy_score(true, predict)
            wf1 = f1_score(true, predict, average="weighted")
            if i == 0:
                bacc_knn.append(balanced_acc)
                wf1_knn.append(wf1)
            else: 
                bacc_log.append(balanced_acc)
                wf1_log.append(wf1)

    bacc_knn = np.array(bacc_knn)
    bacc_log = np.array(bacc_log)
    wf1_knn = np.array(wf1_knn)
    wf1_log = np.array(wf1_log)
    print('wf1_knn', np.mean(wf1_knn), np.std(wf1_knn))
    print('acc_knn', np.mean(bacc_knn), np.std(bacc_knn))
    print('wf1_log', np.mean(wf1_log), np.std(wf1_log))
    print('acc_log', np.mean(bacc_log), np.std(bacc_log))