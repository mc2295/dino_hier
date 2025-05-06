import os
import numpy as np
from sklearn.neighbors import KNeighborsClassifier
import pandas as pd
from sklearn.linear_model import LogisticRegression
from pathlib import Path
def perform_knn(train_data, train_labels, test_data, test_labels, save_dir, filenames,fold):
    # Define a range of values for n_neighbors to search
    n_neighbors_values = [1, 20]
    # n_neighbors_values = [1, 2, 5, 10, 20, 50, 100, 500]
    # n_neighbors_values = [1, 2, 3, 4, 5] # -> for testing
    metrics_dict = {}
    os.makedirs(save_dir, exist_ok=True, mode=0o777)
    os.chmod(save_dir, 0o777)
    test_predictions_all = []

    for n_neighbors in n_neighbors_values:
        # Initialize a KNeighborsClassifier with the current n_neighbors
        knn = KNeighborsClassifier(n_neighbors=n_neighbors)

        # Fit the KNN classifier to the training data
        knn.fit(train_data, train_labels)

        # Predict labels for the test data
        test_predictions = knn.predict(test_data)
        test_predictions_all.append(test_predictions)
        # Evaluate the classifier



        df_labels_to_save = pd.DataFrame({"filename": filenames, "label": test_labels, "prediction": test_predictions})
        filename = f"{Path(save_dir).name}_{n_neighbors}nn_labels_and_predictions_fold_{fold}.csv"
        file_path = os.path.join(save_dir, filename)
        # Speichern des DataFrames in der CSV-Datei
        df_labels_to_save.to_csv(file_path, index=False)

    return test_labels, test_predictions_all

def train_and_evaluate_logistic_regression(train_data, train_labels, test_data, test_labels, save_dir,filenames, fold, max_iter=1000):
    # Initialize wandb

    M = train_data.shape[1]
    C = len(np.unique(train_labels))
    l2_reg_coef = 100 / (M * C)

    # Initialize the logistic regression model with L-BFGS solver
    logistic_reg = LogisticRegression(C=1 / l2_reg_coef, max_iter=max_iter, multi_class="multinomial", solver="lbfgs")

    logistic_reg.fit(train_data, train_labels)

    # Evaluate the model on the test data
    test_predictions = logistic_reg.predict(test_data)
    predicted_probabilities = logistic_reg.predict_proba(train_data)

    # auroc = roc_auc_score(test_labels, test_predictions, multi_class='ovr', average='weighted')

    df_labels_to_save = pd.DataFrame({"filename": filenames, "label": test_labels, "prediction": test_predictions})
    filename = f"{Path(save_dir).name}_logreg_labels_and_predictions_fold_{fold}.csv"
    os.makedirs(save_dir, exist_ok=True, mode=0o777)
    os.chmod(save_dir, 0o777)
    file_path = os.path.join(save_dir, filename)
    # Speichern des DataFrames in der CSV-Datei
    df_labels_to_save.to_csv(file_path, index=False)

    predicted_probabilities_df = pd.DataFrame(
        predicted_probabilities, columns=[f"Probability Class {i}" for i in range(predicted_probabilities.shape[1])]
    )
    predicted_probabilities_filename = f"{Path(save_dir).name}_predicted_probabilities_test.csv"
    predicted_probabilities_file_path = os.path.join(save_dir, predicted_probabilities_filename)
    predicted_probabilities_df.to_csv(predicted_probabilities_file_path, index=False)



    return test_labels, test_predictions

