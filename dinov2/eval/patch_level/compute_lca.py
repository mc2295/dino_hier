import numpy as np
import pandas as pd
from nltk.tree import Tree
from typing import Dict, List, Union

def load_hierarchy() -> Tree:
    """Load the cell type hierarchy tree structure."""
    return Tree("root", [
        Tree("lymphopoiesis.", [
            Tree("immature", ["lymphocyte_immature", "lymphoblast"]),
            Tree("lymphocyte_mature", [
                Tree("typical", ["plasma_cell", "lymphocyte_typical"]),
                Tree("lymphocyte_atypical", ["hairy_cell", "lymphocyte_large_granular", "lymphocyte_reactive", "lymphocyte_neoplastic", "lymphocyte_variant"]),
            ])
        ]),
        Tree("erythropoiesis", [
            Tree("immature", ['proerythroblast', "erythroblast"]),
        ]),
        Tree("monopoiesis", [
            Tree("immature", ["monoblast", "promonocyte"]),
            Tree("mature", ["monocyte"])]),
        Tree("granulopoiesis", [
            Tree("immature", [
                "myeloblast",
                Tree("promyelocyte_all", ["promyelocyte", "promyelocyte_bilobed", "fagott_cell"]),
                "myelocyte",
                "metamyelocyte"
            ]),
            Tree("mature", [
                "basophil",
                Tree("neutrophil", ["neutrophil_band", "neutrophil_segmented"]),
                Tree("eosinophil_all", ["eosinophil", "eosinophil_abnormal"])
            ]),
        ]),
        Tree("artifact_0", ["smudge_cell", "artifact", "not_identifiable", "other"]),
        Tree("blast_0", ["blast"]),
        Tree("platelet_0", ["platelet", "thrombocyte_aggregated", "thrombocyte_giant"]),
    ])

def get_lca_length(location1: List[int], location2: List[int]) -> int:
    """Calculate the length of the least common ancestor path."""
    i = 0
    while i < len(location1) and i < len(location2) and location1[i] == location2[i]:
        i += 1
    return i

def compute_lca_mean(y_true: List[str], y_pred: List[str], hierarchy: Tree) -> float:
    """Compute the mean LCA distance between true and predicted labels."""
    total_dist = 0
    count = 0
   
    for a, b in zip(y_true, y_pred):
        if a in {'ART', 'OTH', 'NIF', 'LYA'} or b in {'ART', 'OTH', 'NIF', 'LYA'}:
            continue
        if a == b:
            continue
           
        leaf_values = hierarchy.leaves()
        leaf_index1 = leaf_values.index(a)
        leaf_index2 = leaf_values.index(b)

        location1 = hierarchy.leaf_treeposition(leaf_index1)
        location2 = hierarchy.leaf_treeposition(leaf_index2)

        dist = get_lca_length(location1, location2)
        total_dist += len(location1) - dist
        count += 1
       
    return total_dist / count if count > 0 else 0

def evaluate_model_predictions(dataset_name: str, root_path: str) -> pd.DataFrame:
    """
    Evaluate model predictions for a given dataset.
   
    Parameters
    ----------
    dataset_name : str
        Name of the dataset to evaluate
    root_path : str
        Root path where model evaluation results are stored
       
    Returns
    -------
    pd.DataFrame
        DataFrame containing evaluation results
    """
    # Define model paths
    folder_list = [
        f"{root_path}resnet50/{dataset_name}_eval/",
        f"{root_path}dinov2_vits14/{dataset_name}_eval/",
        f"{root_path}ctranspath/{dataset_name}_eval/",
        f"{root_path}owkin/{dataset_name}_eval/",
        f"{root_path}conch/{dataset_name}_eval/",
        f"{root_path}conchv1.5/{dataset_name}_eval/",
        f"{root_path}h-optimus-0/{dataset_name}_eval/",
        f"{root_path}vits_5M_350k_bs416_0.1ce+new_supcon_mlp_rbc_cyto_4gpu80gb/{dataset_name}_eval/",
    ]
   
    if dataset_name in {"apl_aml", "acevedo"}:
        dic_res = {
            '1NN_mean': [], '1NN_std': [],
            '20NN_mean': [], '20NN_std': [],
            'logreg_mean': [], 'logreg_std': [],
        }
    elif dataset_name in {"bonemarrow", "rbc"}:
        dic_res = {'lca': []}
    else:
        raise ValueError(f"Unsupported dataset: {dataset_name}")
   
    dic_mapping = class_mappings[dataset_name]
    int_to_label = {i: j for j, i in dic_mapping.items()}
   
    # Process each model
    for model_path in folder_list:
        if dataset_name in {"apl_aml", "acevedo"}:
            # Process kNN and logistic regression results
            for met, n_neighbors in [('20NN', 20), ('1NN', 1), ('logreg', None)]:
                file_pattern = f"{dataset_name}_eval_{'knn' if n_neighbors else 'logreg'}_labels_and_predictions_fold_{{}}"
                if n_neighbors:
                    file_pattern += f"_n{n_neighbors}"
                file_pattern += ".csv"
               
                lca_folds = []
                for fold in range(5):
                    pred_file = model_path + file_pattern.format(fold)
                    df = pd.read_csv(pred_file)
                   
                    y_true = [int_to_label[i] for i in df['label'].tolist()]
                    y_pred = [int_to_label[i] for i in df['prediction'].tolist()]
                   
                    hierarchy = load_hierarchy()
                    lca = compute_lca_mean(y_true, y_pred, hierarchy)
                    lca_folds.append(lca)
               
                dic_res[f'{met}_mean'].append(np.mean(lca_folds))
                dic_res[f'{met}_std'].append(np.std(lca_folds))
               
        elif dataset_name in {"bonemarrow", "rbc"}:
            # Process single evaluation file
            pred_file = f"{model_path}{dataset_name}_eval_labels_and_predictions.csv"
            df = pd.read_csv(pred_file)
           
            y_true = [int_to_label[i] for i in df['True Labels'].tolist()]
            y_pred = [int_to_label[i] for i in df['Predicted Labels'].tolist()]
           
            hierarchy = load_hierarchy()
            lca = compute_lca_mean(y_true, y_pred, hierarchy)
            dic_res['lca'].append(lca)
   
    return pd.DataFrame(dic_res)

if __name__ == "__main__":
    root_path = "/lustre/groups/shared/users/peng_marr/DinoBloomv2/"
    dataset_name = "acevedo"  # or "apl_aml", "bonemarrow", "rbc"

    class_mappings = {
        'apl_aml': {'artifact': 0, 'basophil': 1, 'blast': 2, 'eosinophil': 3, 'erythroblast': 4, 'lymphocyte_typical': 5, 'lymphocyte_variant': 6, 'metamyelocyte': 7, 'monocyte': 8, 'myelocyte': 9, 'neutrophil_band': 10, 'neutrophil_segmented': 11, 'plasma_cell': 12, 'promonocyte': 13, 'promyelocyte': 14, 'smudge_cell': 15, 'thrombocyte_aggregated': 16, 'thrombocyte_giant': 17},
        'raabin': {'basophil': 0, 'eosinophil': 1, 'lymphocyte': 2, 'monocyte': 3, 'neutrophil': 4},
        'acevedo': {'basophil': 0, 'eosinophil': 1, 'erythroblast': 2, 'lymphocyte_typical': 3, 'metamyelocyte': 4, 'monocyte': 5, 'myelocyte': 6, 'neutrophil_band': 7, 'neutrophil_segmented': 8, 'promyelocyte': 9},
        'comparisondetector': {"actinomyces": 0,"agc": 1,"asch": 2,"ascus": 3,"candida": 4,"flora": 5,"herps": 6,"hsil": 7,"lsil": 8,"scc": 9,"trichomonas": 10},
        'sipakmed': {"im_Dyskeratotic": 0, "im_Koilocytotic": 1, "im_Metaplastic": 2, "im_Parabasal": 3, "im_Superficial-Intermediate": 4},
        'hicervix': {"ACTINO": 0, "ADC-EMC": 1, "AGC-EMC-NOS": 2, "ASC-H": 3, "CC": 4, "FUNGI": 5, "HSIL": 6, "MPC": 7, "RPC": 8, "ADC": 9, "AGC": 10, "AGC-FN": 11, "ASC-US": 12, "ECC": 13, "HCG": 14, "HSV": 15, "Normal": 16, "SCC": 17, "ADC-ECC": 18, "AGC-ECC-NOS": 19, "AGC-NOS": 20, "Atrophy": 21, "EMC": 22, "LSIL": 23, "PG": 24, "TRI": 25},
        'lbc': {"HSIL": 0, "LSIL": 1, "N": 2, "SCC": 3},
        'rbc': {'01_rounded_rbcs': 0, '02_ovalocytes': 1, '03_fragmented_rbcs': 2, '04_two_overlapping_rbcs': 3, '05_three_overlapping_rbcs': 4, '06_burr_cells': 5, '07_teardrops': 6, '08_angled_cells': 7, '09_borderline_ovalocytes': 8},
        'bonemarrow': {'eosinophil_abnormal': 0, 'artifact': 1, 'basophil': 2, 'blast': 3, 'erythroblast': 4, 'eosinophil': 5, 'fagott_cell': 6, 'hairy_cell': 7, 'smudge_cell': 8, 'lymphocyte_immature': 9, 'lymphocyte_typical': 10, 'metamyelocyte': 11, 'monocyte': 12, 'myeloblast': 13, 'neutrophil_band': 14, 'neutrophil_segmented': 15, 'not_identifiable': 16, 'other': 17, 'proerythroblast': 18, 'plasma_cell': 19, 'promyelocyte': 20},
    }
    
    results_df = evaluate_model_predictions(dataset_name, root_path)
    results_df.to_csv(f'result_lca_{dataset_name}.csv')

