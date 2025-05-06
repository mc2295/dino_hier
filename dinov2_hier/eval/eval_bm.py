
from dinov2_hier.eval.utils.feature_extraction import save_features_and_labels, get_data
from dinov2_hier.eval.utils.k_fold import create_stratified_folds, mean_bs_dicts
from dinov2_hier.eval.utils.return_model import get_models, get_transforms
from dinov2_hier.eval.utils.dataset import PathImageDataset, CustomImageDataset
from dinov2_hier.eval.utils.classification_tasks import perform_knn, train_and_evaluate_logistic_regression
from dinov2_hier.eval.utils.eval_classification import eval_classification
from dinov2_hier.eval.utils.eval_hier import compute_hier_metric
from torch.utils.data import DataLoader

import os
from pathlib import Path
import numpy as np
import pandas as pd

# BM
bm_path_train = "path/to/bm_train.csv"
bm_path_test = "path/to/bm_test.csv"




model_list = [

    'path/to/model/checkpoints'

]

model_arch_list =  ['dinov2_vitl14'] 

for model_path, model_name in zip(model_list, model_arch_list):



    result_dir = 'results/'+ Path(model_path).parent.parent.stem
    if not os.path.exists(result_dir):
        os.makedirs(result_dir)

    transform = get_transforms(model_name,224,model_path=model_path)
    sorted_paths = list(Path(model_path).rglob("*teacher_checkpoint.pth"))



    checkpoint = sorted_paths[0]

    parent_dir=checkpoint.parent 

    print("loading checkpoint: ", checkpoint)
    feature_extractor = get_models(model_name, 224, saved_model_path=checkpoint)



    feature_dir_bm = parent_dir / 'bm_eval' 
    train_dir = os.path.join(feature_dir_bm, "train_data")
    test_dir = os.path.join(feature_dir_bm, "test_data")

    df = pd.read_csv(bm_path_train)
    df_test = pd.read_csv(bm_path_test)

    valid_labels = {'ABE', 'BAS', 'BLA', 'EBO', 'EOS', 'FGC', 'HAC', 'KSC', 'LYT', 'MMZ', 'MON', 'MYB', 'NGB', 'NGS', 'PEB', 'PLM', 'PMO'}

    df = df.loc[df['Label'].isin(valid_labels) ]
    df_test = df_test.loc[df_test['Label'].isin(valid_labels)]

    df = df.sample(10000, random_state=42)
    #df_test = df_test.sample(500, random_state=42)
    train_dataset = CustomImageDataset(df, transform=transform)
    test_dataset = CustomImageDataset(df_test, transform=transform)

    train_dataloader = DataLoader(train_dataset, batch_size=64, shuffle=False, num_workers=16)
    test_dataloader = DataLoader(test_dataset,  batch_size=64, shuffle=False, num_workers=16)

    save_features_and_labels(feature_extractor, train_dataloader, train_dir,train_dataset,model_name)
    save_features_and_labels(feature_extractor, test_dataloader, test_dir,test_dataset,model_name)

    all_features_train=list(Path(train_dir).glob("*.h5"))
    all_features_test = list(Path(test_dir).glob("*.h5"))
    train_data, train_labels, filenames_train = get_data(all_features_train)
    test_data_all, test_labels_all, test_filenames_all = get_data(all_features_test)






    results_1NN = []
    results_20NN = []
    results_logreg = []
    unique_classes = np.unique(test_labels_all)
    n = len(test_data_all)

    n_bootstrap = 10

    for i in range(n_bootstrap):
        indices = np.random.choice(np.arange(n), size=n, replace=True)
        test_labels = test_labels_all[indices]
        test_data = test_data_all[indices]
        test_filenames = np.array(test_filenames_all)[indices]


        # Create data loaders for the  datasets

        print("data fully loaded")
        logreg_dir = Path(result_dir) /"bm" / "logreg"
        log_reg_label_i, log_reg_pred_i = train_and_evaluate_logistic_regression(
            train_data, train_labels, test_data, test_labels, logreg_dir, test_filenames, i, max_iter=1000
        )

        print("logistic_regression done")


        knn_dir =Path(result_dir) / "bm" / "knn"
        knn_label_i, knn_pred_i = perform_knn(train_data, train_labels, test_data, test_labels, knn_dir, test_filenames,i)
        # knn_pred_i [[pred_1NN], [pred_20NN]]

        print("knn done")


        for pred, label, metric, results in zip(
            [knn_pred_i[0], knn_pred_i[1], log_reg_pred_i],
            [test_labels, test_labels, test_labels],
            ["1NN", "20NN", "logreg"],
            [results_1NN, results_20NN, results_logreg]
        ):

            accuracy, balanced_acc, weighted_f1 = eval_classification(label, pred)
            lca, spearman, hP, hR, hF1, lineage_acc, rare_classes_acc = compute_hier_metric(label, pred, test_data, "bonemarrow")
            results.append({
                'Model Path' : model_path,
                'Metric': metric,
                'wf1': weighted_f1,
                'bAcc' : balanced_acc,
                'acc' : accuracy,
                'LCA': lca,
                'Spearman Ranking' : spearman,
                'Hierarchical Precision' : hP,
                'Hierarchical Recall': hR,
                'Hierarchical F1': hF1,
                'Lineage Accuracy': lineage_acc,
                'Rare Classes Accuracy':rare_classes_acc,
            })


    mean_dict_1NN, std_dict_1NN = mean_bs_dicts(results_1NN)
    mean_dict_20NN, std_dict_20NN = mean_bs_dicts(results_20NN)
    mean_dict_logreg, std_dict_logreg = mean_bs_dicts(results_logreg)

    mean_1NN = pd.DataFrame(mean_dict_1NN, index = [0])
    std_1NN =  pd.DataFrame(std_dict_1NN, index = [0])
    mean_20NN = pd.DataFrame(mean_dict_20NN, index = [1])
    std_20NN =  pd.DataFrame(std_dict_20NN, index = [1])
    mean_logreg = pd.DataFrame(mean_dict_logreg, index = [2])
    std_logreg =  pd.DataFrame(std_dict_logreg, index = [2])

    mean_1NN.to_csv(result_dir + '/bm/mean_1NN.csv')
    std_1NN.to_csv(result_dir + '/bm/bs_1NN.csv')
    mean_20NN.to_csv(result_dir + '/bm/mean_20NN.csv')
    std_20NN.to_csv(result_dir + '/bm/bs_20NN.csv')
    mean_logreg.to_csv(result_dir + '/bm/mean_logreg')
    std_logreg.to_csv(result_dir + '/bm/bs_logreg.csv')

    all_result_mean = pd.concat([mean_1NN, mean_20NN, mean_logreg])
    all_result_mean.to_csv(result_dir + '/bm/all_result_mean.csv')
    all_result_std = pd.concat([std_1NN, std_20NN, std_logreg])
    all_result_std.to_csv(result_dir + '/bm/all_result_bs.csv')


