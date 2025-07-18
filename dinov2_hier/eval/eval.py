
from dinov2_hier.eval.utils.feature_extraction import save_features_and_labels, get_data
from dinov2_hier.eval.utils.k_fold import create_stratified_folds, mean_std_dicts
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

acevedo_path = 'path/to/acevedo/images'


model_list = [

    'path/to/model/checkpoints'
]
model_arch_list =  ['dinov2_vitl14']

for model_path, model_name in zip(model_list, model_arch_list):
    

    result_dir = 'results/'+ Path(model_path).parent.parent.stem
#    if not os.path.exists(result_dir):
#        os.makedirs(result_dir)

    transform = get_transforms(model_name,224,model_path=model_path)
    sorted_paths = list(Path(model_path).rglob("*teacher_checkpoint.pth"))
    print(model_path, sorted_paths)


    checkpoint = sorted_paths[0]

    parent_dir=checkpoint.parent 

    print("loading checkpoint: ", checkpoint)
    feature_extractor = get_models(model_name, 224, saved_model_path=checkpoint)

    ### Acevedo

    feature_dir_ace = parent_dir / 'acevedo_eval' 
    print(str(feature_dir_ace) + '/features.npy') 

    dataset_ace = PathImageDataset(acevedo_path, transform=transform, filetype='.jpg',img_size=(224,224))
    dataloader_ace = DataLoader(dataset_ace, batch_size=64, shuffle=False, num_workers=16)

    save_features_and_labels(feature_extractor, dataloader_ace, feature_dir_ace, len(dataset_ace),model_name)



    all_features=list(feature_dir_ace.glob("*.h5"))

    data,labels,filenames = get_data(all_features)


    #np.save(str(feature_dir_ace) + '/features.npy', data)
    #np.save( str(feature_dir_ace) + '/labels.npy', labels)
    #with open(str(feature_dir_ace) + "/filenames.txt", "w") as output:
    #    output.write(str(filenames))

    #data = np.load(str(feature_dir_ace) + '/features.npy')
    #labels = np.load(str(feature_dir_ace) + '/labels.npy')
    #my_file = open(str(feature_dir_ace) + "/filenames.txt", "r") 
    #filenames = my_file.read() 
    #filenames = filenames.split()

    folds=create_stratified_folds(labels)



    results_1NN = []
    results_20NN = []
    results_logreg = []

    for i, (train_indices, test_indices) in enumerate(folds):
        assert not set(train_indices) & set(test_indices), "There are common indices in train and test lists."

        train_data=data[train_indices]
        train_labels=labels[train_indices]
        

        test_data=data[test_indices]
        test_labels=labels[test_indices]
        test_features = data[test_indices]
        test_filenames=np.array(filenames)[test_indices]
        # Create data loaders for the  datasets

        print("data fully loaded")
        logreg_dir = Path(result_dir) /"ace" / "logreg"
        log_reg_label_i, log_reg_pred_i = train_and_evaluate_logistic_regression(
            train_data, train_labels, test_data, test_labels, logreg_dir, test_filenames,i, max_iter=1000
        )


        print("logistic_regression done")


        knn_dir =Path(result_dir) / "ace" / "knn"
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
            lca, spearman, hP, hR, hF1, lineage_acc, rare_classes_acc = compute_hier_metric(label, pred, test_features, "acevedo")
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

    mean_dict_1NN, std_dict_1NN = mean_std_dicts(results_1NN)

    mean_dict_20NN, std_dict_20NN = mean_std_dicts(results_20NN)
    mean_dict_logreg, std_dict_logreg = mean_std_dicts(results_logreg)

    mean_1NN = pd.DataFrame(mean_dict_1NN, index = [0])
    std_1NN =  pd.DataFrame(std_dict_1NN, index = [0])
    mean_20NN = pd.DataFrame(mean_dict_20NN, index = [1])
    std_20NN =  pd.DataFrame(std_dict_20NN, index = [1])
    mean_logreg = pd.DataFrame(mean_dict_logreg, index = [2])
    std_logreg =  pd.DataFrame(std_dict_logreg, index = [2])

    mean_1NN.to_csv(result_dir + '/ace/mean_1NN.csv')
    std_1NN.to_csv(result_dir + '/ace/std_1NN.csv')
    mean_20NN.to_csv(result_dir + '/ace/mean_20NN.csv')
    std_20NN.to_csv(result_dir + '/ace/std_20NN.csv')
    mean_logreg.to_csv(result_dir + '/ace/mean_logreg')
    std_logreg.to_csv(result_dir + '/ace/std_logreg.csv')

    all_result_mean = pd.concat([mean_1NN, mean_20NN, mean_logreg])
    all_result_mean.to_csv(result_dir + '/ace/all_result_mean.csv')
    all_result_std = pd.concat([std_1NN, std_20NN, std_logreg])
    all_result_std.to_csv(result_dir + '/ace/all_result_std.csv')

